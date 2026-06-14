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

fn createStridesFromShape(allocator: std.mem.Allocator, shape: []const usize) ![]const usize {
    const strides = try allocator.alloc(usize, shape.len);
    defer allocator.free(strides);

    const numberOfElements = shape.len;
    var i = numberOfElements;

    var growingProduct: usize = 1;
    while (i > 0) {
        i -= 1;
        strides[i] = growingProduct;
        growingProduct *= shape[i];
    }

    return strides;
}

pub fn Tensor(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        shape: []const usize,
        strides: []usize,
        device_id: types.Device,
        data: []T,

        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !@This() {
            var element_count: usize = 1;

            if (shape.len == 0) {
                element_count = 0;
            } else {
                for (shape) |dimension_size| {
                    element_count *= dimension_size;
                }
            }

            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const strides = try createStridesFromShape(allocator, shape);

            const owned_strides = try allocator.dupe(usize, strides);
            errdefer allocator.free(owned_strides);

            var data_slice: []T = undefined;

            data_slice = try allocator.alloc(T, element_count);
            @memset(data_slice, 0);
            errdefer allocator.free(data_slice);

            return @This(){
                .allocator = allocator,
                .shape = owned_shape,
                .strides = owned_strides,
                .device_id = types.Device.CPU,
                .data = data_slice,
            };
        }

        pub fn initWithFill(allocator: std.mem.Allocator, shape: []const usize, fill_value: T) !@This() {
            var element_count: usize = 1;

            if (shape.len == 0) {
                element_count = 0;
            } else {
                for (shape) |dimension_size| {
                    element_count *= dimension_size;
                }
            }

            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const strides = try createStridesFromShape(allocator, shape);

            const owned_strides = try allocator.dupe(usize, strides);
            errdefer allocator.free(owned_strides);

            var data_slice: []T = undefined;

            data_slice = try allocator.alloc(T, element_count);
            @memset(data_slice, fill_value);
            errdefer allocator.free(data_slice);

            return @This(){
                .allocator = allocator,
                .shape = owned_shape,
                .strides = owned_strides,
                .device_id = types.Device.CPU,
                .data = data_slice,
            };
        }

        pub fn initWithStrides(allocator: std.mem.Allocator, shape: []const usize, strides: []usize, device_id: types.Device, initial_data: ?[]T, initial_fill: ?T) !@This() {
            var element_count: usize = 1;
            const fill_value = if (initial_fill != null) initial_fill.? else 0;

            if (shape.len == 0 and initial_data == null) {
                element_count = 0;
            } else {
                for (shape) |dimension_size| {
                    element_count *= dimension_size;
                }
            }

            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const owned_strides = try allocator.dupe(usize, strides);
            errdefer allocator.free(owned_strides);

            const element_size = @sizeOf(T);
            const expected_byte_size = element_count * element_size;

            var data_slice: []T = undefined;

            if (initial_data) |provided_data| {
                if (provided_data.len != expected_byte_size / element_size) {
                    std.log.err("Provided data size ({d}) does not match expected size ({d}) from shape ({any}) and type {s}.", .{ provided_data.len, expected_byte_size, shape, @typeName(T) });
                    return error.DataSizeMismatch;
                }
                data_slice = try allocator.dupe(T, provided_data);
                errdefer allocator.free(data_slice);
            } else {
                data_slice = try allocator.alloc(T, element_count);
                @memset(data_slice, fill_value);
                errdefer allocator.free(data_slice);
            }

            return @This(){
                .allocator = allocator,
                .shape = owned_shape,
                .strides = owned_strides,
                .device_id = device_id,
                .data = data_slice,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
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

        pub fn add(self: *@This(), other: @This()) !void {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return errors.Error.ShapeMismatch;
            }
            if (!std.mem.eql(usize, self.strides, other.strides)) {
                return errors.Error.StrideMismatch;
            }

            for (other.data, 0..) |value_to_add, i| {
                self.data[i] += value_to_add;
            }
        }

        pub fn fill(self: *@This(), fill_value: T) void {
            var i: usize = 0;
            while (i < self.data.len) {
                self.data[i] = fill_value;
                i += 1;
            }
        }
    };
}
