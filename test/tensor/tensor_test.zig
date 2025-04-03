const std = @import("std");
const zigtensor = @import("zigtensor");

const Tensor = zigtensor.Tensor;
const DType = zigtensor.DType;
const Device = zigtensor.Device;

test "expect tensor shape to be {2,3}" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ =  gpa.deinit();

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor(f32).initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        null,
    );
    defer t.deinit();

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 6);
}

test "expect tensor slice with initial data" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var data = [_]f32{1,2,3,4,5,6};

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor(f32).initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        &data,
    );
    defer t.deinit();

    const actual_slice: []f32 = try t.slice();

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 6);
    try std.testing.expectEqualSlices(f32, data[0..], actual_slice);

}

test "tensor-tensor element-wise addition" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var data = [_]f32{1,2,3,4,5,6};

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor(f32).initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        &data,
    );
    defer t.deinit();

    const t2 = try Tensor(f32).initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        &data,
    );

    t.add(t2);

    const expected_data = [_]f32{1,4,6,8,10,12};
    const actual_data: []f32 = try t.slice();

    std.testing.expectEqualSlices(expected_data[0..], actual_data);
}
