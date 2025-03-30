const std = @import("std");

const type_tests = @import("./core/types_test.zig");

test "expect true to be true" {
    try std.testing.expect(true == true);
}
