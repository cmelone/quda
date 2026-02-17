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

} // namespace quda
