const std = @import("std");
const zigtensor = @import("zigtensor");

const Tensor = zigtensor.Tensor;
const ops = zigtensor.ops;

test "expect tensor to be sum of two tensors" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    const allocator = gpa.allocator();
    var strides = [_]usize{ 3, 1 };
    var t1 = try Tensor(f32).initCpu(
        allocator,
        &[_]usize{ 2, 3 },
        &strides,
        zigtensor.Device.CPU,
        &data,
        null,
    );
    defer t1.deinit();

    var data2 = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var strides2 = [_]usize{ 3, 1 };
    var t2 = try Tensor(f32).initCpu(
        allocator,
        &[_]usize{ 2, 3 },
        &strides2,
        zigtensor.Device.CPU,
        &data2,
        null,
    );
    defer t2.deinit();

    var t3 = try ops.add(f32, allocator, &t1, &t2);
    defer t3.deinit();

    const s1 = try t1.slice();
    const s2 = try t2.slice();
    const s3 = try t3.slice();

    try std.testing.expect(s3.len == s1.len);
    try std.testing.expect(s3.len == s2.len);
    try std.testing.expect(s3.len == 6);
    try std.testing.expect(s3[0] == 2);

    for (s3, 0..) |item, index| {
        try std.testing.expect(item == s1[index] + s2[index]);
    }
}
