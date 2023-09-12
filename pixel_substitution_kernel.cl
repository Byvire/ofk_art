__kernel void substitutePixels(
                               __global int* size,  // length 1. Remaining inputs are length size[0].
                               // output_band[i] = input_band[substitutions[i]].
                               // So of course all values in substitutions need to be in range(0, size[0]).
                               __global int* substitutions,
                               // Input band, e.g. just the Red pixel values in an image.
                               __global uchar* input_band,
                               // Output buffer for substituted pixel values.
                               __global uchar* output_band) {
  for (int index = get_global_id(0);
       index < size[0];
       index += get_local_size(0) * get_num_groups(0)) {
    output_band[index] = input_band[substitutions[index]];
  }
}
