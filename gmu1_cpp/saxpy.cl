__kernel void vector_saxpy(__global int *y, const __global int *x, int alpha, int vector_size)
{
  int gid = (int) get_global_id(0);
  if (gid < vector_size) {
    y[gid] += x[gid] * alpha;
  }
}

__kernel void vector_mul(__global int *x, int alpha, int vector_size)
{
  int gid = (int) get_global_id(0);
  if (gid < vector_size) {
    x[gid] *= alpha;
  }
}

__kernel void vector_add(__global int *y, const __global int *x, int vector_size)
{
  int gid = (int) get_global_id(0);
  if (gid < vector_size) {
    y[gid] += x[gid];
  }
}
