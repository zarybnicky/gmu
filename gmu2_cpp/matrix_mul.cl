__kernel void matrix_mul_basic(__global int *global_A, __global int *global_B, __global int *global_C, int A_w, int A_h, int B_w)
{
  int global_x = (int)get_global_id(0);
  int global_y = (int)get_global_id(1);

  int B_h = A_w;
  int C_w = B_w;
  int C_h = A_h;

  int reg_C = 0;
  if (global_x < C_w && global_y < C_h) {
    for (int i = 0; i < A_w; i++) {
      reg_C += global_A[i + global_y * A_w] * global_B[global_x + i * B_w];
    }
    global_C[global_x + global_y * C_w] = reg_C;
  }
}

__kernel void matrix_mul_local(__global int *global_A, __global int *global_B, __global int *global_C, int A_w, int A_h, int B_w, __local int *local_A, __local int *local_B)
{
    int global_x = (int)get_global_id(0);
    int global_y = (int)get_global_id(1);
    int local_x = (int)get_local_id(0);
    int local_y = (int)get_local_id(1);
    int local_w = (int)get_local_size(0);
    int local_h = (int)get_local_size(1);

    int B_h = A_w;
    int C_w = B_w;
    int C_h = A_h;
    int reg_C = 0;

    //===========================================================================================  
    /* ======================================================
    * TODO
    * doplnit telo kernelu - s pouzitim sdilene pameti
    * ======================================================= */

    for (int i = 0; i < A_w / local_w; i++) {
      local_A[local_y * local_w + local_x] = global_y < A_h ? global_A[global_y * A_w + local_x + local_w * i] : 0;
      local_B[local_y * local_w + local_x] = global_x < B_w ? global_B[global_x + B_w * (local_y + local_h * i)] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int j = 0; j < local_h; j++) {
        reg_C += local_A[local_y * local_w + j] * local_B[local_h * j + local_x];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_y < C_h && global_x < C_w) {
      global_C[global_y * C_w + global_x] = reg_C;
    }
}
