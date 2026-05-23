const std = @import("std");
const types = @import("../types.zig");
const errors = @import("../errors.zig");

pub fn getMnistData() {
    const gpa = std.heap.DebugAllocator(.{});
    var client: std.http.Client = {
        .allocator = gpa
    }
}
