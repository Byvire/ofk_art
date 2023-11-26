int getRowIndex(int num_columns, int pixel_index) {
  // num_columns is __global dimensions[0], if indexing into the input photo.
  // Else dimensions[0] // factor.
  // Converts flat-array index to row number.
  // dimensions = [width, height].
  // pixel_index is index of a pixel in the flattened image data.
  // pixel_index = row_ix * width + col_ix
  return pixel_index / num_columns;
}

int getColIndex(int num_columns, int pixel_index) {
  // Converts flat-array index to column number.
  return pixel_index % num_columns;
}


int getPixelIndexOrNeg(int num_columns, int num_rows, int row_ix, int col_ix) {
  // Converts row and column numbers to flat-array index, or -1 if the row or
  // column is out-of-bounds.
  if (row_ix < 0 || row_ix >= num_rows || col_ix < 0 || col_ix >= num_columns) {
    return -1;
  }
  return row_ix * num_columns + col_ix;
}


__kernel void downsizeImage(
                          // [width, height] of input photo.
                          // Other photo data is given in flat arrays of size width*height.
                          __global int* dimensions,
                          // photo input data, in LAB color coordinates. Read only.
                          __global uchar* lab_band0,
                          __global uchar* lab_band1,
                          __global uchar* lab_band2,

                          // photo output data, also in LAB color coordinates. Write only.
                          __global uchar* out_band0,
                          __global uchar* out_band1,
                          __global uchar* out_band2,
                          // Single int. How many times smaller the output photo will be than the input.
                          // For example, if this is 3, and the input dimensions are 300 x 600, then the
                          // output dimensions are assumed to be 100 x 200. (And buffers should be sized
                          // accordingly.) In this case, each output pixel would be the average of 9
                          // input pixels.
                          __global int* factor) {
  // By rounding down here, we cut off the right/bottom portion of the image and
  // avoid handling the condition where some of the input pixels corresponding
  // to an output pixel are out of bounds.
  int out_num_cols = dimensions[0] / factor[0];
  int out_num_rows = dimensions[1] / factor[0];


    for (int index = get_global_id(0);
         index < dimensions[0] * dimensions[1] / (factor[0] * factor[0]);
         index += get_local_size(0) * get_num_groups(0)) {

      int out_col = getColIndex(out_num_cols, index);
      int out_row = getRowIndex(out_num_cols, index);

      uint total_band0 = 0;
      uint total_band1 = 0;
      uint total_band2 = 0;
      for (int dx = 0; dx < factor[0]; dx += 1) {
        for (int dy = 0; dy < factor[0]; dy += 1) {
          int in_index = getPixelIndexOrNeg(dimensions[0], dimensions[1],
                                            out_row * factor[0] + dy, out_col * factor[0] + dx);
          if (in_index < 0) {
            continue;   // Should never happen.
          }
          total_band0 += lab_band0[in_index];
          total_band1 += lab_band1[in_index];
          total_band2 += lab_band2[in_index];
        }
      }
      out_band0[index] = (uchar) (total_band0 / (factor[0] * factor[0]));
      out_band1[index] = (uchar) (total_band1 / (factor[0] * factor[0]));
      out_band2[index] = (uchar) (total_band2 / (factor[0] * factor[0]));
    }

}
