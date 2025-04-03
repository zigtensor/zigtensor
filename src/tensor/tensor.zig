const std = @import("std");
const types = @import("../types.zig");
const errors = @import("../errors.zig");

fn sizeOfDType(dtype: types.DType) usize {
    return switch (dtype) {
        .f32 => @sizeOf(f32),
        .i32 => @sizeOf(i32),
        .bool => @sizeOf(bool),
    };
}

pub fn Tensor(comptime T: type) type {
    return struct {
        allocator: *std.mem.Allocator,
        shape: []const usize,
        strides: []usize,
        device_id: types.Device,
        data: []T,

        pub fn initCpu (
            allocator: *std.mem.Allocator,
            shape: []const usize,
            strides: []usize,
            device_id: types.Device,
            initial_data: ?[]T
        ) !@This() {
            var element_count: usize = 1;

            if (shape.len == 0 and initial_data == null) {
                element_count = 0;
            } else {
                for (shape) |dimension_size| {
                    element_count *= dimension_size;
                }
            }

            const element_size = @sizeOf(T);
            const expected_byte_size = element_count * element_size;

            var data_slice: []T = undefined;

            if (initial_data) |provided_data| {
                if (provided_data.len != expected_byte_size / element_size) {
                    std.log.err("Provided data size ({d}) does not match expected size ({d}) from shape ({d}) and type {s}.",
                        .{ provided_data.len, expected_byte_size, shape, @typeName(T)});
                    return error.MismatchedDataSize;
                }
                data_slice = try allocator.dupe(T, provided_data);
            } else {
                data_slice = try allocator.alloc(T, element_count);
                @memset(std.mem.sliceAsBytes(data_slice), 0);
            }

            return @This() {
                .allocator = allocator,
                .shape = shape,
                .strides = strides,
                .device_id = device_id,
                .data = data_slice,
            };
        }
        
        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.data);
        }

        pub fn slice(self: *@This()) ![]T {
            if (self.device_id == types.Device.CPU) {
                return self.data;
            } else {
                // Temporarily while no GPU allocation is implemented
                return errors.Error.InvalidDevice;
            }
        }
    };
}
