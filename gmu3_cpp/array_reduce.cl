__kernel void global_atomic_reduce_sum(__global int *a, __global int *result, int array_size)
{
  int global_x = (int)get_global_id(0);
  atomic_add(result, global_x < array_size ? a[global_x] : 0);
}

__kernel void local_atomic_reduce_sum(__global int *a, __global int *result, int array_size)
{
  int global_x = (int)get_global_id(0);
  __local int local_result;
  if (get_local_id(0) == 0)
    local_result = 0;
  atomic_add(&local_result, global_x < array_size ? a[global_x] : 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    atomic_add(result, local_result);
  }
}

__kernel void local_reduce_sum(__global int *a, __global int *result, int array_size, __local int *tmp_a)
{
  int global_x = (int)get_global_id(0);
  int local_x = (int)get_local_id(0);
  int local_w = (int)get_local_size(0);

  tmp_a[local_x] = global_x < array_size ? a[global_x] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 1; i < local_w; i <<= 1) {
    if (local_x % (i << 1) == 0) {
      tmp_a[local_x] += tmp_a[local_x + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (get_local_id(0) == 0) {
    atomic_add(result, tmp_a[0]);
  }
}

__kernel void local_naive_reduce_sum(__global int *a, __global int *result, int array_size, __local int *tmp_a)
{
    int global_x = (int)get_global_id(0);
    int local_x = (int)get_local_id(0);
    int local_w = (int)get_local_size(0);

    // naplneni lokalni pameti daty pro skupinu
    tmp_a[local_x] = global_x < array_size ? a[global_x] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    // i <<= 1 je to same jak i *= 2
    for (int i = 1; i < local_w; i <<= 1) {
        if ((local_x % (i << 1) == 0) && (global_x + i < array_size))
            tmp_a[local_x] += tmp_a[local_x + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // pricteni do globalni promenne
    if (local_x == 0) atom_add(result, tmp_a[local_x]);
}




