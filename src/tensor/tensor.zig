const std = @import("std");
const types = @import("../types.zig");

fn sizeOfDType(dtype: types.DType) usize {
    return switch (dtype) {
        .f32 => @sizeOf(f32),
        .i32 => @sizeOf(i32),
        .bool => @sizeOf(bool),
    };
}

pub const Tensor = struct {
    allocator: *std.mem.Allocator,
    shape: []const usize,
    strides: []usize,
    dtype: types.DType,
    device_id: types.Device,
    data: []u8,

    pub fn initCpu (
        allocator: *std.mem.Allocator,
        shape: []const usize,
        strides: []usize,
        dtype: types.DType,
        device_id: types.Device,
        initial_data: ?[]u8
    ) !Tensor {
        var element_count: usize = 1;

        if (shape.len == 0 and initial_data == null) {
            element_count = 0;
        } else {
            for (shape) |dimension_size| {
                element_count *= dimension_size;
            }
        }

        const element_size = sizeOfDType(dtype);
        const expected_byte_size = element_count * element_size;

        var data_slice: []u8 = undefined;

        if (initial_data) |provided_data| {
            if (provided_data.len != expected_byte_size) {
                std.log.err("Provided data size ({d}) does not match expected size ({d}) from shape ({d}) and dtype {s}.", .{ provided_data.len, expected_byte_size, shape, @tagName(dtype)});
                return error.MismatchedDataSize;
            }
            data_slice = try allocator.dupe(u8, provided_data);
        } else {
            data_slice = try allocator.alloc(u8, expected_byte_size);
            @memset(data_slice, 0);
        }

        return Tensor{
            .allocator = allocator,
            .shape = shape,
            .strides = strides,
            .dtype = dtype,
            .device_id = device_id,
            .data = data_slice,
        };
    }
    
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }
};
