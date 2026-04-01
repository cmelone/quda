#include <gauge_field.h>
#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <worker.h>
#include <tunable_nd.h>
#include <kernels/dslash_coarse.cuh>
#include <shmem_helper.cuh>
#include <dslash_quda.h>
#include <dslash_shmem.h>
#include <multigrid.h>

#ifdef QUDA_MNEME_ANNOTATIONS
#include "mneme/MnemeAnnotation.hpp"
#endif

// BEGIN COPY FROM dslash_coarse.cuh
namespace quda {

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
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    int coord[4];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    if (!thread_dir || target::is_host()) {

      //Forward gather - compute fwd offset for spinor fetch
#pragma unroll
      for (int d0 = 0; d0 < Arg::nDim; d0 += Arg::dim_stride) { // loop over dimension
        int d = d0 + thread_dim;
        const int fwd_idx = linkIndexHop(coord, arg.dim, d, arg.nFace);

        if (arg.commDim[d] && is_boundary(coord, d, 1, arg)) {
          if constexpr (doHalo<Arg::type>()) {
            int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

#pragma unroll
            for (int color_local = 0; color_local < Mc; color_local++) { //Color row
              int c_row = color_block + color_local; // global color index
              int row = s_row * Arg::nColor + c_row;
#pragma unroll
              for (int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
                for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                  int col = s_col * Arg::nColor + c_col + color_offset;
                  if (!Arg::dagger)
                    out[color_local]
                      = cmac(arg.Y(d + 4, parity, x_cb, row, col), arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col + color_offset), out[color_local]);
                  else
                    out[color_local]
                      = cmac(arg.Y(d, parity, x_cb, row, col),
                             arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col + color_offset), out[color_local]);
                }
              }
            }
          }
        } else if constexpr (doBulk<Arg::type>()) {
#pragma unroll
          for (int color_local = 0; color_local < Mc; color_local++) { //Color row
            int c_row = color_block + color_local; // global color index
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for (int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
              for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local]
                    = cmac(arg.Y(d + 4, parity, x_cb, row, col),
                           arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col + color_offset),
                           out[color_local]);
                else
                  out[color_local]
                    = cmac(arg.Y(d, parity, x_cb, row, col),
                           arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col + color_offset),
                           out[color_local]);
              }
            }
          }
        }

      } // nDim
    }

    if (thread_dir || target::is_host()) {

      //Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
      for (int d0 = 0; d0 < Arg::nDim; d0 += Arg::dim_stride) {
        const int d = d0 + thread_dim;
        const int back_idx = linkIndexHop(coord, arg.dim, d, -arg.nFace);

        if (arg.commDim[d] && is_boundary(coord, d, 0, arg)) {
          if constexpr (doHalo<Arg::type>()) {
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
#pragma unroll
            for (int color_local = 0; color_local < Mc; color_local++) {
              int c_row = color_block + color_local;
              int row = s_row * Arg::nColor + c_row;
#pragma unroll
              for (int s_col = 0; s_col < Arg::nSpin; s_col++)
#pragma unroll
                for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                  int col = s_col * Arg::nColor + c_col + color_offset;
                  if (!Arg::dagger)
                    out[color_local]
                      = cmac(conj(arg.Y.Ghost(d, 1 - parity, ghost_idx, col, row)),
                             arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col,
                                            c_col + color_offset),
                             out[color_local]);
                  else
                    out[color_local]
                      = cmac(conj(arg.Y.Ghost(d + 4, 1 - parity, ghost_idx, col, row)),
                             arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col,
                                            c_col + color_offset),
                             out[color_local]);
                }
            }
          }
        } else if constexpr (doBulk<Arg::type>()) {
          const int gauge_idx = back_idx;
#pragma unroll
          for (int color_local = 0; color_local < Mc; color_local++) {
            int c_row = color_block + color_local;
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for (int s_col = 0; s_col < Arg::nSpin; s_col++)
#pragma unroll
              for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local]
                    = cmac(conj(arg.Y(d, 1 - parity, gauge_idx, col, row)),
                           arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col + color_offset),
                           out[color_local]);
                else
                  out[color_local]
                    = cmac(conj(arg.Y(d + 4, 1 - parity, gauge_idx, col, row)),
                           arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col + color_offset),
                           out[color_local]);
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

    // M is number of colors per thread
#pragma unroll
    for (int color_local = 0; color_local < Mc; color_local++) { //Color out
      int c = color_block + color_local; // global color index
      int row = s * Arg::nColor + c;
#pragma unroll
      for (int s_col = 0; s_col < Arg::nSpin; s_col++) //Spin in
#pragma unroll
        for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color in
          //Factor of kappa and diagonal addition now incorporated in X
          int col = s_col * Arg::nColor + c_col + color_offset;
          if (!Arg::dagger) {
            out[color_local]
              = cmac(arg.X(0, parity, x_cb, row, col), arg.inB[src_idx](spinor_parity, x_cb, s_col, c_col + color_offset),
                     out[color_local]);
          } else {
            out[color_local] = cmac(conj(arg.X(0, parity, x_cb, col, row)),
                                    arg.inB[src_idx](spinor_parity, x_cb, s_col, c_col + color_offset), out[color_local]);
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
        for (int d = 1; d < Arg::dim_stride; d++) { // get remaining forward gathers (if any)
          // 4-way 1,2,3  (stride = 4)
          // 2-way 1      (stride = 2)
          out += cache.load_z(target::thread_idx().z + d * 2 + 0);
        }

#pragma unroll
        for (int d = 0; d < Arg::dim_stride; d++) { // get all backward gathers
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

  // BEGIN kernel of interest
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
        const int vector_site_width = device::warp_size() / Arg::color_stride; // number of sites per warp

        x_cb = target::block_idx().x * (target::block_dim().x / Arg::color_stride)
          + warp_id * (device::warp_size() / Arg::color_stride) + lane_id % vector_site_width;
        color_offset = lane_id / vector_site_width;
      }

      int src_idx = src_parity % arg.n_src;
      int parity = (arg.nParity == 2) ? (src_parity / arg.n_src) : arg.parity;

      // z thread dimension is (( s*(Nc/Mc) + color_block )*dim_thread_split + dim)*2 + dir
      constexpr int Mc = CoarseDslashParams<Arg>::Mc;
      int dir = sMd & 1;
      int sMdim = sMd >> 1;
      int dim = sMdim % Arg::dim_stride;
      int sM = sMdim / Arg::dim_stride;
      int s = sM / (Arg::nColor / Mc);
      int color_block = (sM % (Arg::nColor / Mc)) * Mc;

      typename CoarseDslashParams<Arg>::array_t out {};

      if (Arg::dslash) {
        applyDslash<Mc>(out, dim, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);
        target::dispatch<dim_collapse>(out, dir, dim, *this);
      }

      if (doBulk<Arg::type>() && Arg::clover && dir == 0 && dim == 0)
        applyClover<Mc>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);

      if (dir == 0 && dim == 0) {
        const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;

        // reduce down to the first group of column-split threads
        out = warp_combine<Arg::color_stride>(out);

#pragma unroll
        for (int color_local = 0; color_local < Mc; color_local++) {
          int c = color_block + color_local; // global color index
          if (color_offset == 0) {
            // if not halo we just store, else we accumulate
            if (doBulk<Arg::type>())
              arg.out[src_idx](my_spinor_parity, x_cb, s, c) = out[color_local];
            else
              arg.out[src_idx](my_spinor_parity, x_cb, s, c) += out[color_local];
          }
        }
      }
    }
  };

  // END kernel of interest

  // END COPY FROM dslash_coarse.cuh

  // CNM: DSlashCoarse-specific kernel entry: keep the launch symbol local to this file
  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel3D_launch_bounds<Arg>()) CoarseKernel3D(Arg arg)
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(arg);
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void>
    __launch_bounds__(device::get_default_kernel3D_launch_bounds<Arg>()) CoarseKernel3D()
  {
    Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
  }
  // END DSlashCoarse-specific kernel entry

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, int Nc, bool dslash, bool clover, bool dagger,
            DslashType type>
  class DslashCoarse : public TunableKernel3D
  {
    static constexpr int nDim = 4;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const GaugeField &Y;
    const GaugeField &X;
    const double kappa;
    const int parity;
    const int nParity;
    const ColorSpinorField &halo;

    const int max_color_col_stride = 8;
    mutable int color_col_stride;
    mutable int dim_threads;

    long long flops() const
    {
      return ((dslash * 2 * nDim + clover * 1) * (8 * Ns * Nc * Ns * Nc) - 2 * Ns * Nc) * nParity
        * (long long)out.VolumeCB() * out.size();
    }
    long long bytes() const
    {
      return (dslash || clover) * out.Bytes() + dslash * 8 * inA.Bytes() + clover * inB.Bytes()
        + (nParity * (dslash * Y.Bytes() * Y.VolumeCB() / (2 * Y.Stride()) + clover * X.Bytes() / 2)) * out.size();
    }

    unsigned int sharedBytesPerThread() const
    {
      return (sizeof(complex<compute_prec<Float>>) * colors_per_thread(Nc, dim_threads));
    }
    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions
    unsigned int minThreads() const { return color_col_stride * X.VolumeCB(); }

    /**
       @param Helper function to check that the present launch parameters are valid
    */
    bool checkParam(const TuneParam &param) const
    {
      return ((color_col_stride == 1 || minThreads() % (unsigned)device::warp_size() == 0)
              && // active threads must be a multiple of the warp
              (color_col_stride == 1 || param.block.x % device::warp_size() == 0)
              &&                                        // block must be a multiple of the warp
              Nc % color_col_stride == 0 &&             // number of colors must be divisible by the split
              param.grid.x < device::max_grid_size(0)); // ensure the resulting grid size valid
    }

    bool advanceColorStride(TuneParam &param) const
    {
      bool valid = false;

      while (param.aux.x < max_color_col_stride) {
        param.aux.x *= 2;
        color_col_stride = param.aux.x;
        param.grid.x
          = (minThreads() + param.block.x - 1) / param.block.x; // grid size changed since minThreads has been updated
        valid = checkParam(param);
        if (valid) break;
      }

      if (!valid) {
        // reset color column stride if too large or not divisible
        param.aux.x = 1;
        color_col_stride = param.aux.x;
        param.grid.x
          = (minThreads() + param.block.x - 1) / param.block.x; // grid size changed since minThreads has been updated
      }

      return valid;
    }

    bool advanceDimThreads(TuneParam &param) const
    {
      bool rtn;
      if (2 * param.aux.y <= nDim && param.block.x * param.block.y * dim_threads * 2 <= device::max_threads_per_block()) {
        param.aux.y *= 2;
        rtn = true;
      } else {
        param.aux.y = 1;
        rtn = false;
      }

      dim_threads = param.aux.y;
      // need to reset z-block/grid size/shared_bytes since dim_threads has changed
      resizeStep(step_y, 2 * dim_threads);
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::initTuneParam(param);

      return rtn;
    }

#ifndef QUDA_FAST_COMPILE_DSLASH
    bool advanceAux(TuneParam &param) const { return advanceColorStride(param) || advanceDimThreads(param); }
#else
    bool advanceAux(TuneParam &) const { return false; }
#endif

    void initTuneParam(TuneParam &param) const
    {
      color_col_stride = 1;
      dim_threads = 1;
      resizeStep(step_y, 2 * dim_threads); // 2 is forwards/backwards
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::initTuneParam(param);
      param.aux = make_int4(color_col_stride, dim_threads, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      color_col_stride = 1;
      dim_threads = 1;
      resizeStep(step_y, 2 * dim_threads); // 2 is forwards/backwards
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::defaultTuneParam(param);
      param.aux = make_int4(color_col_stride, dim_threads, 1, 1);

      // ensure that the default x block size is divisible by the warpSize
      param.block.x = device::warp_size();
      param.grid.x = (minThreads() + param.block.x - 1) / param.block.x;
      param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
    }

  public:
    DslashCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                 cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, double kappa,
                 int parity, MemoryLocation *halo_location, const ColorSpinorField &halo) :
      TunableKernel3D(out[0], out.SiteSubset() * out.size(), 1),
      out(out),
      inA(inA),
      inB(inB),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      halo(halo),
      color_col_stride(-1)
    {
      strcpy(aux, (std::string("policy_kernel,") + aux).c_str());
      strcat(aux, comm_dim_partitioned_string());

      switch(type) {
      case DSLASH_INTERIOR: strcat(aux,",interior"); break;
      case DSLASH_EXTERIOR: strcat(aux,",exterior"); break;
      case DSLASH_FULL:     strcat(aux,",full"); break;
      }

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      if (doHalo<type>()) {
        char label[15] = ",halo=";
        for (int dim=0; dim<4; dim++) {
          for (int dir=0; dir<2; dir++) {
            label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : halo_location[2*dim+dir] == Device ? '1' : halo_location[2*dim+dir] == Host ? '2' : '3';
          }
        }
        label[14] = '\0';
        strcat(aux,label);
      }

      setRHSstring(aux, inA.size());
#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      apply(device::get_default_stream());
    }

    template <int color_stride, int dim_stride, bool native = true>
    using Arg
      = DslashCoarseArg<dslash, clover, dagger, type, color_stride, dim_stride, Float, yFloat, ghostFloat, Ns, Nc, native>;

    template <int color_stride, int dim_stride> inline void launch_coarse(const TuneParam &tp, const qudaStream_t &stream)
    {
      // previously launched generic kernel3d via launch_device<CoarseDslash>...
      // here we launch through a coarseDslash specific kernel entry (CoarseKernel3D) so the launch is not generic
      using ArgType = Arg<color_stride, dim_stride>;
      auto arg = ArgType(out, inA, inB, Y, X, (Float)kappa, parity, halo);

      // see include/tunable_nd.h:438-443
      // set y and z because we bypass the generic TunableKernel3D::launch_device helper
      arg.threads.y = vector_length_y;
      arg.threads.z = vector_length_z;

    // from include/kernel_helper.h:58-62
    #ifdef JITIFY
      const kernel_t kernel(nullptr, "CoarseKernel3D");
    #else
      // now binds directly to CoarseKernel3D entrypoint instead of the generic Kernel3D
      const kernel_t kernel(reinterpret_cast<const void *>(CoarseKernel3D<CoarseDslash, ArgType, false>), "CoarseKernel3D");
    #endif

      // launch the kernel via the generic launch_device helper (to enable tuning)
      // we still keep the kernel entry local to this file
      TunableKernel::launch_device<CoarseDslash, false>(kernel, tp, stream, arg);
    }

    inline void launch_coarse(const TuneParam &tp, const qudaStream_t &stream)
    {
      // original implementation had the switch/case in apply()
      // the mapping is the same, but relocated here for the local kernel launches
      switch (tp.aux.y) { // dimension gather parallelisation
      case 1:
        switch (tp.aux.x) { // this is color_col_stride
        case 1: launch_coarse<1, 1>(tp, stream); break;
#ifndef QUDA_FAST_COMPILE_DSLASH
        case 2: launch_coarse<2, 1>(tp, stream); break;
        case 4: launch_coarse<4, 1>(tp, stream); break;
        case 8: launch_coarse<8, 1>(tp, stream); break;
#endif
        default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
        }
        break;
#ifndef QUDA_FAST_COMPILE_DSLASH
      case 2:
        switch (tp.aux.x) { // this is color_col_stride
        case 1: launch_coarse<1, 2>(tp, stream); break;
        case 2: launch_coarse<2, 2>(tp, stream); break;
        case 4: launch_coarse<4, 2>(tp, stream); break;
        case 8: launch_coarse<8, 2>(tp, stream); break;
        default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
        }
        break;
      case 4:
        switch (tp.aux.x) { // this is color_col_stride
        case 1: launch_coarse<1, 4>(tp, stream); break;
        case 2: launch_coarse<2, 4>(tp, stream); break;
        case 4: launch_coarse<4, 4>(tp, stream); break;
        case 8: launch_coarse<8, 4>(tp, stream); break;
        default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
        }
        break;
#endif
      default: errorQuda("Invalid dimension thread splitting %d", static_cast<int>(tp.aux.y));
      }
    }

    // apply() no longer performs the generic launch, we delegate to launch_coarse()
    void apply(const qudaStream_t &stream)
    {
      const TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      color_col_stride = tp.aux.x;
      dim_threads = tp.aux.y;
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      if (!checkParam(tp)) errorQuda("Invalid launch param");

      if (out.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not enabled");
      } else {
        checkNative(out[0], inA[0], inB[0], Y, X);

#ifdef QUDA_MNEME_ANNOTATIONS
        // ============================================
        // MNEME ANNOTATIONS - Applied Before CoarseKernel3D Launch
        // ============================================

        // Annotate input spinor fields (should match closely as inputs)
        for (auto i = 0u; i < inA.size(); i++) {
          if (inA[i].data()) {
            mneme::annotate<Float>(static_cast<Float*>(inA[i].data()), mneme::Metadata{
                .threshold      = 1e-7,
                .threshold_kind = mneme::ThresholdKind::Absolute,
                .norm           = mneme::Norm::L2,
                .tag            = "CoarseKernel3D.inA.rhs" + std::to_string(i),
            });
          }

          if (inB[i].data()) {
            mneme::annotate<Float>(static_cast<Float*>(inB[i].data()), mneme::Metadata{
                .threshold      = 1e-7,
                .threshold_kind = mneme::ThresholdKind::Absolute,
                .norm           = mneme::Norm::L2,
                .tag            = "CoarseKernel3D.inB.rhs" + std::to_string(i),
            });
          }
        }

        // Annotate coarse gauge field Y (should be exact)
        if (Y.data()) {
          mneme::annotate<yFloat>(static_cast<yFloat*>(Y.data()), mneme::Metadata{
              .threshold      = 1e-7,
              .threshold_kind = mneme::ThresholdKind::Absolute,
              .norm           = mneme::Norm::Linf,
              .tag            = "CoarseKernel3D.gauge_Y",
          });
        }

        // Annotate coarse clover field X (should be exact)
        if (X.data()) {
          mneme::annotate<yFloat>(static_cast<yFloat*>(X.data()), mneme::Metadata{
              .threshold      = 1e-7,
              .threshold_kind = mneme::ThresholdKind::Absolute,
              .norm           = mneme::Norm::Linf,
              .tag            = "CoarseKernel3D.clover_X",
          });
        }

        // Annotate output spinor fields (looser tolerance for accumulated results)
        for (auto i = 0u; i < out.size(); i++) {
          if (out[i].data()) {
            mneme::annotate<Float>(static_cast<Float*>(out[i].data()), mneme::Metadata{
                .threshold      = 1e-6,
                .threshold_kind = mneme::ThresholdKind::Absolute,
                .norm           = mneme::Norm::L2,
                .tag            = "CoarseKernel3D.output.rhs" + std::to_string(i),
            });
          }
        }
#endif // QUDA_MNEME_ANNOTATIONS

        launch_coarse(tp, stream);
      }
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
  };

  template <template <class, class, class, int, bool, bool, DslashType> class D, typename Float, typename yFloat,
            typename ghostFloat, bool dagger, int coarseColor, bool use_mma, int nVec>
  inline void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                          cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                          double kappa, int parity, bool dslash, bool clover, DslashType type,
                          MemoryLocation *halo_location, const ColorSpinorField &halo)
  {
    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch inA = %d, out = %d", inA.FieldOrder(), out.FieldOrder());

    if (inA.Nspin() != 2) errorQuda("Unsupported number of coarse spins %d", inA.Nspin());

    constexpr int coarseSpin = 2;

    if (dslash) {
      if (clover) {
        switch (type) {
        case DSLASH_FULL: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, true, DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity,
                                                                                   halo_location, halo);
          break;
        }
        case DSLASH_EXTERIOR: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, true, DSLASH_EXTERIOR> dslash(out, inA, inB, Y, X, kappa,
                                                                                       parity, halo_location, halo);
          break;
        }
        case DSLASH_INTERIOR: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, true, DSLASH_INTERIOR> dslash(out, inA, inB, Y, X, kappa,
                                                                                       parity, halo_location, halo);
          break;
        }
        default: errorQuda("Dslash type %d not instantiated", type);
        }

      } else { // plain dslash

        switch (type) {
        case DSLASH_FULL: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, false, DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity,
                                                                                    halo_location, halo);
          break;
        }
        case DSLASH_EXTERIOR: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, false, DSLASH_EXTERIOR> dslash(out, inA, inB, Y, X, kappa,
                                                                                        parity, halo_location, halo);
          break;
        }
        case DSLASH_INTERIOR: {
          D<Float, yFloat, ghostFloat, coarseSpin, true, false, DSLASH_INTERIOR> dslash(out, inA, inB, Y, X, kappa,
                                                                                        parity, halo_location, halo);
          break;
        }
        default: errorQuda("Dslash type %d not instantiated", type);
        }
      }
    } else {

      if (type == DSLASH_EXTERIOR) errorQuda("Cannot call halo on pure clover kernel");
      if (clover) {
        D<Float, yFloat, ghostFloat, coarseSpin, false, true, DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity,
                                                                                  halo_location, halo);
      } else {
        errorQuda("Unsupported dslash=false clover=false");
      }
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

  enum class DslashCoarsePolicy {
    DSLASH_COARSE_BASIC,                   // stage both sends and recvs in host memory using memcpys
    DSLASH_COARSE_ZERO_COPY_PACK,          // zero copy write pack buffers
    DSLASH_COARSE_ZERO_COPY_READ,          // zero copy read halos in dslash kernel
    DSLASH_COARSE_ZERO_COPY,               // full zero copy
    DSLASH_COARSE_SHMEM,                   // non overlapping shmem exchange
    DSLASH_COARSE_SHMEM_OVERLAP,           // overlapping shmem exchange
    DSLASH_COARSE_GDR_SEND,                // GDR send
    DSLASH_COARSE_GDR_RECV,                // GDR recv
    DSLASH_COARSE_GDR,                     // full GDR
    DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV, // zero copy write and GDR recv
    DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ, // GDR send and zero copy read
    DSLASH_COARSE_POLICY_DISABLED
  };

  template <template <class, class, class, int, bool, bool, DslashType> class D, bool dagger, int coarseColor,
            bool use_mma_, int nVec>
  struct DslashCoarseLaunch {

    constexpr static bool use_mma = use_mma_;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const ColorSpinorField &halo;
    const GaugeField &Y;
    const GaugeField &X;
    double kappa;
    int parity;
    bool dslash;
    bool clover;
    const int *commDim;
    const QudaPrecision halo_precision;
    static constexpr bool enable_coarse_shmem_overlap() { return false; }

    DslashCoarseLaunch(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                       cvector_ref<const ColorSpinorField> &inB, const ColorSpinorField &halo, const GaugeField &Y,
                       const GaugeField &X, double kappa, int parity, bool dslash, bool clover, const int *commDim,
                       QudaPrecision halo_precision) :
      out(out),
      inA(inA),
      inB(inB),
      halo(halo),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      dslash(dslash),
      clover(clover),
      commDim(commDim),
      halo_precision(halo_precision == QUDA_INVALID_PRECISION ? Y.Precision() : halo_precision)
    {
    }

    /**
       @brief Execute the coarse dslash using the given policy
     */
    inline void operator()(DslashCoarsePolicy policy)
    {
      if (inA[0].data() == out[0].data()) errorQuda("Aliasing pointers");

      // check all precisions match
      QudaPrecision precision = checkPrecision(out[0], inA[0], inB[0]);
      checkPrecision(Y, X);

      // check all locations match
      checkLocation(out[0], inA[0], inB[0], Y, X);

      int comm_sum = 4;
      if (commDim) for (int i=0; i<4; i++) comm_sum -= (1-commDim[i]);
      if (comm_sum != 4 && comm_sum != 0) errorQuda("Unsupported comms %d", comm_sum);
      bool comms = comm_sum;
      int shmem = 0;

      MemoryLocation pack_destination[2 * QUDA_MAX_DIM]; // where we will pack the ghost buffer to
      MemoryLocation halo_location[2 * QUDA_MAX_DIM];    // where we load the halo from
      bool gdr_send = false;
      bool gdr_recv = false;
      if (policy == DslashCoarsePolicy::DSLASH_COARSE_SHMEM || policy == DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP) {
        for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) {
          pack_destination[i] = Shmem;
          halo_location[i] = Device;
        }
        shmem = 1;
      } else {
        for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) {
          pack_destination[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK
                                 || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
                                 || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ?
            Host :
            Device;
          halo_location[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ
                              || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
                              || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ?
            Host :
            Device;
        }
        gdr_send = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR
                    || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ?
          true :
          false;
        gdr_recv = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR
                    || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ?
          true :
          false;
      }
      // disable peer-to-peer if doing a zero-copy policy (temporary)
      bool p2p_enabled = comm_peer2peer_enabled_global();
      if (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV
          || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ)
        comm_enable_peer2peer(false);

      if (policy != DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP) {
        ////////////////
        /// NO OVERLAP

        if (dslash && comm_partitioned() && comms) {
          const int nFace = 1;
          halo.exchangeGhost((QudaParity)(inA.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? (1 - parity) : 0), nFace,
                             dagger, pack_destination, halo_location, gdr_send, gdr_recv, halo_precision, shmem, inA);
        }

        if (dslash::aux_worker) dslash::aux_worker->apply(device::get_default_stream());

        if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
          if (Y.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("Y Precision %d not supported", Y.Precision());
          if (halo_precision != QUDA_DOUBLE_PRECISION)
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                      precision, Y.Precision());
          ApplyCoarse<D, double, double, double, dagger, coarseColor, use_mma, nVec>(
            out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_FULL : DSLASH_INTERIOR, halo_location,
            halo);
#else
          errorQuda("Double precision multigrid has not been enabled");
#endif
        } else if (precision == QUDA_SINGLE_PRECISION) {
          if (Y.Precision() == QUDA_SINGLE_PRECISION) {
            if (halo_precision == QUDA_SINGLE_PRECISION) {
              ApplyCoarse<D, float, float, float, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_FULL : DSLASH_INTERIOR,
                halo_location, halo);
            } else {
              errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                        precision, Y.Precision());
            }
          } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
            if (halo_precision == QUDA_HALF_PRECISION) {
              ApplyCoarse<D, float, short, short, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_FULL : DSLASH_INTERIOR,
                halo_location, halo);
            } else if (halo_precision == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
              ApplyCoarse<D, float, short, int8_t, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_FULL : DSLASH_INTERIOR,
                halo_location, halo);
#else
              errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
            } else {
              errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                        precision, Y.Precision());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
          } else {
            errorQuda("Unsupported precision %d", Y.Precision());
          }
        } else {
          errorQuda("Unsupported precision %d", Y.Precision());
        }
      } else if constexpr (DslashCoarseLaunch<D, dagger, coarseColor, use_mma, nVec>::enable_coarse_shmem_overlap()) {
// OVERLAP
#ifdef NVSHMEM_COMMS
        if (dslash && comm_partitioned() && comms) {
          const int nFace = 1;
          shmem += 2;
          halo.exchangeGhost((QudaParity)(inA.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? (1 - parity) : 0), nFace,
                             dagger, pack_destination, halo_location, gdr_send, gdr_recv, halo_precision, shmem, inA);
        }
        // INTERIOR
        if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
          if (Y.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("Y Precision %d not supported", Y.Precision());
          if (halo_precision != QUDA_DOUBLE_PRECISION)
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                      precision, Y.Precision());
          ApplyCoarse<D, double, double, double, dagger, coarseColor, use_mma, nVec>(
            out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_INTERIOR : DSLASH_INTERIOR,
            halo_location, halo);
#else
          errorQuda("Double precision multigrid has not been enabled");
#endif
        } else if (precision == QUDA_SINGLE_PRECISION) {
          if (Y.Precision() == QUDA_SINGLE_PRECISION) {
            if (halo_precision == QUDA_SINGLE_PRECISION) {
              ApplyCoarse<D, float, float, float, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_INTERIOR : DSLASH_INTERIOR,
                halo_location, halo);
            } else {
              errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                        precision, Y.Precision());
            }
          } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
            if (halo_precision == QUDA_HALF_PRECISION) {
              ApplyCoarse<float, short, short, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_INTERIOR : DSLASH_INTERIOR,
                halo_location, halo);
            } else if (halo_precision == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
              ApplyCoarse<D, float, short, int8_t, dagger, coarseColor, use_mma, nVec>(
                out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_INTERIOR : DSLASH_INTERIOR,
                halo_location, halo);
#else
              errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
            } else {
              errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                        precision, Y.Precision());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
          } else {
            errorQuda("Unsupported precision %d", Y.Precision());
          }
        } else {
          errorQuda("Unsupported precision %d", Y.Precision());
        }
        if (dslash::aux_worker) dslash::aux_worker->apply(device::get_default_stream());

        if (dslash && comm_partitioned() && comms) {
          quda::dslash::shmem_signal_wait_all();

          // exterior
          if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
            if (Y.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("Y Precision %d not supported", Y.Precision());
            if (halo_precision != QUDA_DOUBLE_PRECISION)
              errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                        precision, Y.Precision());
            ApplyCoarse<D, double, double, double, dagger, coarseColor, use_mma, nVec>(
              out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_EXTERIOR : DSLASH_EXTERIOR,
              halo_location, halo);
#else
	errorQuda("Double precision multigrid has not been enabled");
#endif
          } else if (precision == QUDA_SINGLE_PRECISION) {
            if (Y.Precision() == QUDA_SINGLE_PRECISION) {
              if (halo_precision == QUDA_SINGLE_PRECISION) {
                ApplyCoarse<D, float, float, float, dagger, coarseColor, use_mma, nVec>(
                  out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_EXTERIOR : DSLASH_INTERIOR,
                  halo_location, halo);
              } else {
                errorQuda("Halo precision %d not supported with field precision %d and link precision %d",
                          halo_precision, precision, Y.Precision());
              }
            } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
          if (halo_precision == QUDA_HALF_PRECISION) {
            ApplyCoarse<D, float, short, short, dagger, coarseColor, use_mma, nVec>(
              out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_EXTERIOR : DSLASH_EXTERIOR,
              halo_location, halo);
          } else if (halo_precision == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
            ApplyCoarse<D, float, short, int8_t, dagger, coarseColor, use_mma, nVec>(
              out, inA, inB, Y, X, kappa, parity, dslash, clover, comms ? DSLASH_EXTERIOR : DSLASH_EXTERIOR,
              halo_location, halo);
#else
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
          } else {
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision,
                      precision, Y.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
        } else {
          errorQuda("Unsupported precision %d", Y.Precision());
        }
          } else {
            errorQuda("Unsupported precision %d", Y.Precision());
          }
        }
#else
        errorQuda("NVSHMEM policy called but NVSHMEM not enabled.");
#endif
      }
      if (dslash && comm_partitioned() && comms) inA[0].bufferIndex = (1 - inA[0].bufferIndex);

      comm_enable_peer2peer(p2p_enabled); // restore the p2p state
    }
  };

  template <typename Launch>
  class DslashCoarsePolicyTune : public Tunable {

    static inline bool dslash_init = false;
    static inline int first_active_policy = static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED);
    // string used as a tunekey to ensure we retune if the dslash policy env changes
    static inline char policy_string[TuneKey::aux_n] = {};
    static inline std::vector<DslashCoarsePolicy> policies = {static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED), DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED};

    static void enable_policy(DslashCoarsePolicy p) { policies[static_cast<std::size_t>(p)] = p; }

    // Helper to get policy name for debugging
    static const char* getPolicyName(DslashCoarsePolicy p) {
      switch(p) {
        case DslashCoarsePolicy::DSLASH_COARSE_BASIC: return "BASIC";
        case DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK: return "ZERO_COPY_PACK";
        case DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ: return "ZERO_COPY_READ";
        case DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY: return "ZERO_COPY";
        case DslashCoarsePolicy::DSLASH_COARSE_SHMEM: return "SHMEM";
        case DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP: return "SHMEM_OVERLAP";
        case DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND: return "GDR_SEND";
        case DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV: return "GDR_RECV";
        case DslashCoarsePolicy::DSLASH_COARSE_GDR: return "GDR";
        case DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV: return "ZERO_COPY_PACK_GDR_RECV";
        case DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ: return "GDR_SEND_ZERO_COPY_READ";
        case DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED: return "DISABLED";
        default: return "UNKNOWN";
      }
    }

   Launch &dslash;

   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
   static constexpr bool enable_coarse_shmem_overlap = Launch::enable_coarse_shmem_overlap();

 public:
   DslashCoarsePolicyTune(Launch &dslash) : dslash(dslash)
   {
      if (!dslash_init) {

	static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_COARSE_POLICY");

	if (dslash_policy_env) { // set the policies to tune for explicitly
	  std::stringstream policy_list(dslash_policy_env);

	  int policy_;
	  while (policy_list >> policy_) {
	    DslashCoarsePolicy dslash_policy = static_cast<DslashCoarsePolicy>(policy_);

	    // check this is a valid policy choice
	    if ( (dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND ||
            dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV ||
		        dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR ||
            dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV ||
		        dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) && !comm_gdr_enabled() ) {
	      errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", static_cast<int>(dslash_policy));
	    }

	    enable_policy(dslash_policy);
	    first_active_policy = policy_ < first_active_policy ? policy_ : first_active_policy;
	    if (policy_list.peek() == ',') policy_list.ignore();
	  }
	  if(first_active_policy == static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED)) errorQuda("No valid policy found in QUDA_ENABLE_DSLASH_COARSE_POLICY");
	} else {
          first_active_policy = 0;
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_BASIC);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY);
          if (comm_nvshmem_enabled()) {
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_SHMEM);
            if constexpr (enable_coarse_shmem_overlap) enable_policy(DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP);
          }
          if (comm_gdr_enabled()) {
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ);
          }
        }

        // Runtime blacklist: disable specific policies via environment variable
        // Usage: export QUDA_DISABLE_DSLASH_COARSE_POLICY="6,7,8,9,10"  # Disable all GDR
        //        export QUDA_DISABLE_DSLASH_COARSE_POLICY="6,7,8"       # Disable GDR_SEND, GDR_RECV, GDR only
        static char *disable_policy_env = getenv("QUDA_DISABLE_DSLASH_COARSE_POLICY");
        if (disable_policy_env) {
          std::stringstream disable_list(disable_policy_env);
          int policy_;
          if (getVerbosity() >= QUDA_VERBOSE) {
            printfQuda("PolicyTune: Disabling policies from QUDA_DISABLE_DSLASH_COARSE_POLICY=%s\n", disable_policy_env);
          }
          while (disable_list >> policy_) {
            if (policy_ >= 0 && policy_ < static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED)) {
              policies[policy_] = DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED;
              if (getVerbosity() >= QUDA_VERBOSE) {
                printfQuda("  - Disabled policy %d = %s\n",
                          policy_, getPolicyName(static_cast<DslashCoarsePolicy>(policy_)));
              }
            }
            if (disable_list.peek() == ',') disable_list.ignore();
          }
        }

        // construct string specifying which policies have been enabled
        strcat(policy_string, ",pol=");
        for (int i = 0; i < (int)DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED; i++) {
          strcat(policy_string, (int)policies[i] == i ? "1" : "0");
        }

        // Print summary of enabled policies
        if (getVerbosity() >= QUDA_VERBOSE) {
          printfQuda("PolicyTune: Enabled policies for DslashCoarse:\n");
          for (int i = 0; i < (int)DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED; i++) {
            if ((int)policies[i] == i) {
              printfQuda("  [%d] %s\n", i, getPolicyName(static_cast<DslashCoarsePolicy>(i)));
            }
          }
        }

        dslash_init = true;
      }

      strcpy(aux, "policy,");
      if (dslash.dslash) strcat(aux, "dslash");
      strcat(aux, dslash.clover ? "clover," : ",");
      strcat(aux, dslash.inA.AuxString().c_str());
      strcat(aux, ",gauge_prec=");

      char prec_str[16];
      i32toa(prec_str, dslash.Y.Precision());
      strcat(aux, prec_str);
      strcat(aux, ",halo_prec=");
      i32toa(prec_str, dslash.halo_precision);
      strcat(aux, prec_str);
      strcat(aux, comm_dim_partitioned_string(dslash.commDim));
      strcat(aux, comm_dim_topology_string());
      strcat(aux, comm_config_string()); // and change in P2P/GDR will be stored as a separate tunecache entry
      strcat(aux, policy_string);        // any change in policies enabled will be stored as a separate entry

      int comm_sum = 4;
      if (dslash.commDim)
        for (int i = 0; i < 4; i++) comm_sum -= (1 - dslash.commDim[i]);
      strcat(aux, comm_sum ? ",full" : ",interior");

      if (Launch::use_mma) { strcat(aux, ",mma"); }
      strcat(aux, ",n_rhs=");
      char rhs_str[16];
      i32toa(rhs_str, dslash.out.size() * dslash.out[0].Nvec());
      strcat(aux, rhs_str);

#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      // before we do policy tuning we must ensure the kernel
      // constituents have been tuned since we can't do nested tuning
      if (!tuned()) {
        disableProfileCount();
	for (auto &i : policies) if(i!= DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) dslash(i);
	enableProfileCount();
	setPolicyTuning(true);
      }
   }

   virtual ~DslashCoarsePolicyTune() { setPolicyTuning(false); }

   inline void apply(const qudaStream_t &)
   {
     TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

     if (tp.aux.x >= (int)policies.size()) errorQuda("Requested policy that is outside of range");
     if (policies[tp.aux.x] == DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED ) errorQuda("Requested policy is disabled");

     // DEBUG: Print which policy we're executing
     if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
       int policy_idx = static_cast<int>(policies[tp.aux.x]);
       printfQuda("PolicyTune: Executing policy %d = %s\n", policy_idx, getPolicyName(policies[tp.aux.x]));
       fflush(stdout);
     }

     dslash(policies[tp.aux.x]);

     if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
       int policy_idx = static_cast<int>(policies[tp.aux.x]);
       printfQuda("PolicyTune: Successfully completed policy %d = %s\n", policy_idx, getPolicyName(policies[tp.aux.x]));
       fflush(stdout);
     }
   }

   bool advanceAux(TuneParam &param) const
   {
    while ((unsigned)param.aux.x < policies.size()-1) {
      param.aux.x++;
      if (policies[param.aux.x] != DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) {
        // DEBUG: Print which policy we're about to test
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
          int policy_idx = static_cast<int>(policies[param.aux.x]);
          printfQuda("PolicyTune: Advancing to policy %d = %s\n", policy_idx, getPolicyName(policies[param.aux.x]));
          fflush(stdout);
        }
        return true;
      }
    }
    param.aux.x = 0;
    return false;
   }

   bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

   void initTuneParam(TuneParam &param) const
   {
     Tunable::initTuneParam(param);
     param.aux = make_int4(first_active_policy, 0, 0, 0);
   }

   void defaultTuneParam(TuneParam &param) const
   {
     Tunable::defaultTuneParam(param);
     param.aux = make_int4(first_active_policy, 0, 0, 0);
   }

   TuneKey tuneKey() const { return TuneKey(dslash.inA.VolString().c_str(), typeid(*this).name(), aux); }

   long long flops() const {
     int nDim = 4;
     int Ns = dslash.inA.Nspin();
     int Nc = dslash.inA.Ncolor() / dslash.inA[0].Nvec();
     int nParity = dslash.inA.SiteSubset();
     long long volumeCB = dslash.inA.VolumeCB();
     return ((dslash.dslash * 2 * nDim + dslash.clover * 1) * (8 * Ns * Nc * Ns * Nc) - 2 * Ns * Nc) * nParity
       * volumeCB * dslash.out.size() * dslash.out[0].Nvec();
   }

   long long bytes() const {
     int nParity = dslash.inA.SiteSubset();
     return (dslash.dslash || dslash.clover) * dslash.out.Bytes() + dslash.dslash * 8 * dslash.inA.Bytes()
       + dslash.clover * dslash.inB.Bytes()
       + (nParity
          * (dslash.dslash * dslash.Y.Bytes() * dslash.Y.VolumeCB() / (2 * dslash.Y.Stride())
             + dslash.clover * dslash.X.Bytes() / 2))
       * dslash.out.size() * dslash.out[0].Nvec();
     // multiply Y by volume / stride to correct for pad
   }
  };

} // namespace quda
