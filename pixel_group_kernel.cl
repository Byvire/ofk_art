
int getRowIndex(__global int* dimensions, int pixel_index) {
  // Converst flat-array index to row number.
  // dimensions = [width, height].
  // pixel_index is index of a pixel in the flattened image data.
  // pixel_index = row_ix * width + col_ix
  return pixel_index / dimensions[0];
}

int getColIndex(__global int* dimensions, int pixel_index) {
  // Converts flat-array index to column number.
  return pixel_index % dimensions[0];
}

int getPixelIndexOrNeg(__global int* dimensions, int row_ix, int col_ix) {
  // Converts row and column numbers to flat-array index, or -1 if the row or
  // column is out-of-bounds.
  if (row_ix < 0 || row_ix >= dimensions[1] || col_ix < 0 || col_ix >= dimensions[0]) {
    return -1;
  }
  return row_ix * dimensions[0] + col_ix;
}

bool groupIsAlive(__global int* groups, int seed_index) {
  // A group is "alive" if its seed pixel still belongs to the group.
  // This helps avoid "unanchored" colors wandering around the image.
  return groups[seed_index] == seed_index;
}

bool distance_within_threshold(__global uchar* lab_band0,
                               __global uchar* lab_band1,
                               __global uchar* lab_band2,
                              int index_a,
                              int index_b,
                              int threshold) {
  // Whether Euclidean distance between the two pixels' colors is within the threshold.
  if (index_a < 0 || index_b < 0) {
    // Avoid segfault (see getPixelIndexOrNeg).
    return false;
  }
  /* return true; */
  int diff0 = ((int) lab_band0[index_a]) - ((int) lab_band0[index_b]);
  int diff1 = ((int) lab_band1[index_a]) - ((int) lab_band1[index_b]);
  int diff2 = ((int) lab_band2[index_a]) - ((int) lab_band2[index_b]);
  return (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) < (threshold * threshold);
}

__kernel void groupPixels(
                          // [width, height] of photo.
                          // Other photo data is given in flat arrays of size width*height.
                          __global int* dimensions,
                          // photo input data, in LAB color coordinates. Read only.
                          __global uchar* lab_band0,
                          __global uchar* lab_band1,
                          __global uchar* lab_band2,

                          // Single value representing threshold of
                          // color-metric-distance for group inclusion.
                          //
                          // If the computed distance between a group's seed
                          // pixel's color and a target pixel's color is within
                          // this threshold, then that target pixel can be a
                          // member of that group.
                          __global int* threshold,

                          // Each pixel has a distinct "priority".
                          // Each pixel is the "seed" of a group. If two groups
                          // want to occupy the same pixel, the group with a
                          // higher-priority seed wins.
                          __global int* priorities,

                          // For each pixel, what group ID does that pixel currently belong to?
                          // A group ID is the index of the seed pixel of that group.
                          // (So on round 0, groups looks like [0, 1, 2, 3, ...].)
                          __global int* groups,
                          // After this iteration, what group ID does each pixel belong to?
                          // Output param.
                          __global int* groups_out) {
  for (int index = get_global_id(0);
       index < dimensions[0] * dimensions[1];
       index += get_local_size(0) * get_num_groups(0)) {

    // Once a group's seed pixel is overwritten, we treat it as low-priority ("dead").
    // Otherwise the edge of a "dead" group could keep propagating even if another group
    // is chasing it and replacing it constantly.
    // (Think of this as FTL signalling to erase a wave that's travelling away at lightspeed.)
    int current_group_priority = groupIsAlive(groups, groups[index]) ? priorities[groups[index]] : -1;
    groups_out[index] = groups[index];  // may get further updates
    int my_row = getRowIndex(dimensions, index);
    int my_col = getColIndex(dimensions, index);

    for (int row_delta = -1; row_delta < 2; row_delta += 1) {
      for (int col_delta = -1; col_delta < 2; col_delta += 1) {
        int neighbor_index = getPixelIndexOrNeg(dimensions, my_row + row_delta, my_col + col_delta);
        if (neighbor_index < 0) {  // row and/or column was out of bounds
          continue;
        }
        // The rule is, a group can only contain pixels close enough to the
        // group's seed pixel's color value.
        if (current_group_priority < priorities[groups[neighbor_index]]
            && groupIsAlive(groups, groups[neighbor_index])
            && distance_within_threshold(lab_band0, lab_band1, lab_band2,
                                         index, groups[neighbor_index], threshold[0])) {
          current_group_priority = priorities[groups[neighbor_index]];
          groups_out[index] = groups[neighbor_index];
        }
      }
    }
  }
}

// For unit testing only.
__kernel void convertIndicesToRowColumnAndBack(
                               __global int* dimensions,
                               __global int* indices_out) {
  for (int index = get_global_id(0);
       index < dimensions[0] * dimensions[1];
       index += get_local_size(0) * get_num_groups(0)) {
    int my_row = getRowIndex(dimensions, index);
    int my_col = getColIndex(dimensions, index);
    indices_out[index] = getPixelIndexOrNeg(dimensions, my_row, my_col);
    /* indices_out[index] = my_row; */
 }
}
