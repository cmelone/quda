#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <shared_memory_cache_helper.h>
#include <kernel.h>
#include <warp_collective.h>
#include <dslash_quda.h>

namespace quda {

  enum DslashType {
    DSLASH_INTERIOR,
    DSLASH_EXTERIOR,
    DSLASH_FULL
  };

#ifdef MULTIGRID_DSLASH_PROMOTE
  template <typename store_t>
  using compute_prec = double;
#else
  template <typename store_t>
  using compute_prec = typename mapper<store_t>::type;
#endif

  // we use two colors per thread unless we have large dim_stride, when we're aiming for maximum parallelism
  constexpr int colors_per_thread(int nColor, int dim_stride) { return (nColor % 2 == 0 && nColor <= 32 && dim_stride <= 2) ? 2 : 1; }

  template <bool dslash_, bool clover_, bool dagger_, DslashType type_, int color_stride_, int dim_stride_, typename Float,
            typename yFloat, typename ghostFloat, int nSpin_, int nColor_, bool native>
  struct DslashCoarseArg : kernel_param<> {
    static constexpr bool dslash = dslash_;
    static constexpr bool clover = clover_;
    static constexpr bool dagger = dagger_;
    static constexpr DslashType type = type_;
    static constexpr int color_stride = color_stride_;
    static constexpr int dim_stride = dim_stride_;

    using real = compute_prec<Float>;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nDim = 4;
    static constexpr int nFace = 1;

    static constexpr QudaFieldOrder csOrder = native ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    static constexpr QudaGaugeFieldOrder gOrder = native ? QUDA_NATIVE_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;

    using G = typename colorspinor::GhostOrder<real, nSpin, nColor, 1, csOrder, Float, ghostFloat>;
    // disable ghost to reduce arg size
    using F = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, csOrder, Float, ghostFloat, true>;
    using GY = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;

    const int_fastdiv n_src;
    F out[MAX_MULTI_RHS];
    F inA[MAX_MULTI_RHS];
    F inB[MAX_MULTI_RHS];
    G halo;
    const GY Y;
    const GY X;
    const real kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;
    int ghostFaceCB[4];

    DslashCoarseArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                    cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real kappa,
                    int parity, const ColorSpinorField &halo) :
      kernel_param(dim3(color_stride * X.VolumeCB(), out.SiteSubset() * out.size(),
                        2 * dim_stride * 2 * (nColor / colors_per_thread(nColor, dim_stride)))),
      n_src(out.size()),
      halo(halo, nFace),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      X0h(((3 - nParity) * out.X(0)) / 2),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out[0].Ndim() == 5 ? out.X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out.VolumeCB() / dim[4])
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->inA[i] = inA[i];
        this->inB[i] = inB[i];
      }
      // ghostFaceCB does not include the batch (5th) dimension at present
      for (int i = 0; i < 4; i++) ghostFaceCB[i] = halo.getDslashConstant().ghostFaceCB[i];
    }
  };

  /**
     @brief Helper function to determine if should halo computation
  */
  template <DslashType type>
  static constexpr bool doHalo() {
    switch(type) {
    case DSLASH_EXTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     @brief Helper function to determine if should interior computation
  */
  template <DslashType type>
  static constexpr bool doBulk() {
    switch(type) {
    case DSLASH_INTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     Applies the coarse dslash on a given parity and checkerboard site index
     /out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)

     @param[in,out] out The result vector
     @param[in] thread_dir Direction
     @param[in] x_cb The checkerboarded site index
     @param[in] src_idx Which src are we working on
     @param[in] parity The site parity
     @param[in] s_row Which spin row are acting on
     @param[in] color_block Which color row are we acting on
     @param[in] color_off Which color column offset are we acting on
     @param[in] arg Arguments
   */
  template <int Mc, typename V, typename Arg>
  __device__ __host__ inline void applyDslash(V &out, int thread_dim, int thread_dir, int x_cb, int src_idx, int parity, int s_row, int color_block, int color_offset, const Arg &arg)
  {
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;
    const auto &Y = arg.Y;
    const auto &halo = arg.halo;
    const auto &inA = arg.inA[src_idx];
    const int row_base = s_row * Arg::nColor + color_block;

    int coord[4];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    if (!thread_dir || target::is_host()) {

      //Forward gather - compute fwd offset for spinor fetch
      for(int d0 = 0; d0 < Arg::nDim; d0 += Arg::dim_stride) { // loop over dimension
        const int d = d0 + thread_dim;
        const int gauge_dir = d + (Arg::dagger ? 0 : 4);

        if (arg.commDim[d] && is_boundary(coord, d, 1, arg) ) {
          if constexpr (doHalo<Arg::type>()) {
            const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
            const int halo_idx = ghost_idx + src_idx * arg.ghostFaceCB[d];

#pragma unroll
            for (int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
              const int spin_col = s_col * Arg::nColor;
#pragma unroll
              for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                const int c = c_col + color_offset;
                const int col = spin_col + c;
                const auto in = halo.Ghost(d, 1, their_spinor_parity, halo_idx, s_col, c);
#pragma unroll
                for (int color_local = 0; color_local < Mc; color_local++) { //Color row
                  const int row = row_base + color_local;
                  out[color_local] = cmac(Y(gauge_dir, parity, x_cb, row, col), in, out[color_local]);
                }
              }
            }
          }
        } else if constexpr (doBulk<Arg::type>()) {
          const int fwd_idx = linkIndexHop(coord, arg.dim, d, arg.nFace);
#pragma unroll
          for (int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
            const int spin_col = s_col * Arg::nColor;
#pragma unroll
            for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
              const int c = c_col + color_offset;
              const int col = spin_col + c;
              const auto in = inA(their_spinor_parity, fwd_idx, s_col, c);
#pragma unroll
              for (int color_local = 0; color_local < Mc; color_local++) { //Color row
                const int row = row_base + color_local;
                out[color_local] = cmac(Y(gauge_dir, parity, x_cb, row, col), in, out[color_local]);
              }
            }
          }
        }

      } // nDim
    }

    if (thread_dir || target::is_host()) {

      //Backward gather - compute back offset for spinor and gauge fetch
      for (int d0 = 0; d0 < Arg::nDim; d0 += Arg::dim_stride) {
        const int d = d0 + thread_dim;
        const int gauge_dir = d + (Arg::dagger ? 4 : 0);

        if (arg.commDim[d] && is_boundary(coord, d, 0, arg)) {
          if constexpr (doHalo<Arg::type>()) {
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
            const int halo_idx = ghost_idx + src_idx * arg.ghostFaceCB[d];
#pragma unroll
            for (int s_col = 0; s_col < Arg::nSpin; s_col++) {
              const int spin_col = s_col * Arg::nColor;
#pragma unroll
              for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                const int c = c_col + color_offset;
                const int col = spin_col + c;
                const auto in = halo.Ghost(d, 0, their_spinor_parity, halo_idx, s_col, c);
#pragma unroll
                for (int color_local = 0; color_local < Mc; color_local++) {
                  const int row = row_base + color_local;
                  out[color_local]
                    = cmac(conj(Y.Ghost(gauge_dir, 1 - parity, ghost_idx, col, row)), in, out[color_local]);
                }
              }
            }
          }
        } else if constexpr (doBulk<Arg::type>()) {
          const int back_idx = linkIndexHop(coord, arg.dim, d, -arg.nFace);
#pragma unroll
          for (int s_col = 0; s_col < Arg::nSpin; s_col++) {
            const int spin_col = s_col * Arg::nColor;
#pragma unroll
            for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
              const int c = c_col + color_offset;
              const int col = spin_col + c;
              const auto in = inA(their_spinor_parity, back_idx, s_col, c);
#pragma unroll
              for (int color_local = 0; color_local < Mc; color_local++) {
                const int row = row_base + color_local;
                out[color_local] = cmac(conj(Y(gauge_dir, 1 - parity, back_idx, col, row)), in, out[color_local]);
              }
            }
          }
        }

      } //nDim
    } // forwards / backwards thread split
  }

  /**
     Applies the coarse clover matrix on a given parity and
     checkerboard site index

     @param out The result out += X * in
     @param X The coarse clover field
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <int Mc, typename V, typename Arg>
  __device__ __host__ inline void applyClover(V &out, const Arg &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset)
  {
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const auto &X = arg.X;
    const auto &inB = arg.inB[src_idx];
    const int row_base = s * Arg::nColor + color_block;

    // M is number of colors per thread
#pragma unroll
    for (int s_col = 0; s_col < Arg::nSpin; s_col++) //Spin in
#pragma unroll
      for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color in
        //Factor of kappa and diagonal addition now incorporated in X
        const int c = c_col + color_offset;
        const int col = s_col * Arg::nColor + c;
        const auto in = inB(spinor_parity, x_cb, s_col, c);
#pragma unroll
        for (int color_local = 0; color_local < Mc; color_local++) { //Color out
          const int row = row_base + color_local;
          if constexpr (!Arg::dagger) {
            out[color_local] = cmac(X(0, parity, x_cb, row, col), in, out[color_local]);
          } else {
            out[color_local] = cmac(conj(X(0, parity, x_cb, col, row)), in, out[color_local]);
          }
        }
      }
  }

  template <bool is_device> struct dim_collapse {
    template <typename T, typename Ftor> void operator()(T &out, int, int, const Ftor &ftor) { out *= -ftor.arg.kappa; }
  };

  template <> struct dim_collapse<true> {
    template <typename T, typename Ftor>
    __device__ __host__ inline void operator()(T &out, int dir, int dim, const Ftor &ftor)
    {
      using Arg = typename Ftor::Arg;
      SharedMemoryCache<T> cache {ftor};
      // only need to write to shared memory if not master thread
      if (dim > 0 || dir) cache.save(out);

      cache.sync(); // recombine the foward and backward results

      if (dir == 0 && dim == 0) {
        // full split over dimension and direction
#pragma unroll
        for (int d=1; d < Arg::dim_stride; d++) { // get remaining forward gathers (if any)
          // 4-way 1,2,3  (stride = 4)
          // 2-way 1      (stride = 2)
          out += cache.load_z(target::thread_idx().z + d * 2 + 0);
        }

#pragma unroll
        for (int d=0; d < Arg::dim_stride; d++) { // get all backward gathers
          out += cache.load_z(target::thread_idx().z + d * 2 + 1);
        }

        out *= -ftor.arg.kappa;
      }
    }
  };

  template <typename Arg> struct CoarseDslashParams {
    static constexpr int Mc = colors_per_thread(Arg::nColor, Arg::dim_stride);
    using array_t = array<complex<typename Arg::real>, Mc>;
    using Ops = KernelOps<SharedMemoryCache<array_t>, op_warp_combine<array_t>>;
  };

  template <typename Arg_> struct CoarseDslash : CoarseDslashParams<Arg_>::Ops {
    using Arg = Arg_;
    const Arg &arg;
    using typename CoarseDslashParams<Arg>::Ops::KernelOpsT;
    template <typename... OpsArgs>
    constexpr CoarseDslash(const Arg &arg, const OpsArgs &...ops) : KernelOpsT(ops...), arg(arg)
    {
    }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb_color_offset, int src_parity, int sMd)
    {
      int x_cb = x_cb_color_offset;
      int color_offset = 0;

      if (target::is_device() && Arg::color_stride > 1) { // on the device we support warp fission of the inner product
        const int lane_id = target::thread_idx().x % device::warp_size();
        const int warp_id = target::thread_idx().x / device::warp_size();
        constexpr int vector_site_width = device::warp_size() / Arg::color_stride; // number of sites per warp

        x_cb = target::block_idx().x * (target::block_dim().x / Arg::color_stride)
          + warp_id * vector_site_width + lane_id % vector_site_width;
        color_offset = lane_id / vector_site_width;
      }

      const int src_idx = src_parity % arg.n_src;
      const int parity = (arg.nParity == 2) ? (src_parity / arg.n_src) : arg.parity;

      // z thread dimension is (( s*(Nc/Mc) + color_block )*dim_thread_split + dim)*2 + dir
      constexpr int Mc = CoarseDslashParams<Arg>::Mc;
      constexpr int n_color_block = Arg::nColor / Mc;
      const int dir = sMd & 1;
      const int sMdim = sMd >> 1;
      const int dim = sMdim % Arg::dim_stride;
      const int sM = sMdim / Arg::dim_stride;
      const int s = sM / n_color_block;
      const int color_block = (sM % n_color_block) * Mc;

      typename CoarseDslashParams<Arg>::array_t out {};

      if constexpr (Arg::dslash) {
        applyDslash<Mc>(out, dim, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);
        target::dispatch<dim_collapse>(out, dir, dim, *this);
      }

      if constexpr (doBulk<Arg::type>() && Arg::clover) {
        if (dir == 0 && dim == 0) applyClover<Mc>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);
      }

      if (dir == 0 && dim == 0) {
        const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;

        // reduce down to the first group of column-split threads
        out = warp_combine<Arg::color_stride>(out);

        if (color_offset == 0) {
#pragma unroll
          for (int color_local = 0; color_local < Mc; color_local++) {
            const int c = color_block + color_local; // global color index
            // if not halo we just store, else we accumulate
            if constexpr (doBulk<Arg::type>())
              arg.out[src_idx](my_spinor_parity, x_cb, s, c) = out[color_local];
            else
              arg.out[src_idx](my_spinor_parity, x_cb, s, c) += out[color_local];
          }
        }
      }
    }
  };

} // namespace quda
