const std = @import("std");

pub fn Set(comptime T: type) type {
    const is_string = comptime isZigString(T);
    return struct {
        const map_type = if (is_string) std.StringHashMap(bool) else std.AutoHashMap(T, bool);
        var map: map_type = undefined;
        const Self = @This();

        pub fn put(_: Self, x: T) std.mem.Allocator.Error!void {
            try map.put(x, true);
        }

        pub fn remove(_: Self, x: T) bool {
            map.remove(x);
        }

        pub fn iterator(_: Self) std.StringHashMap(bool).KeyIterator {
            return map.keyIterator();
        }

        pub fn init(allocator: std.mem.Allocator) Self {
            map = map_type.init(allocator);
            return Self{};
        }

        pub fn deinit(_: Self) void {
            map.deinit();
        }
    };
}

// https://ziggit.dev/t/how-to-check-if-something-is-a-string/5857
// https://github.com/ziglang/zig/commit/d5e21a4f1a2920ef7bbe3c54feab1a3b5119bf77#diff-adfee52549c345d50c3acbd67802c959e2ba7d46f7c747844035b898c8510888L393-L394
/// Returns true if the passed type will coerce to []const u8.
/// Any of the following are considered strings:
/// ```
/// []const u8, [:S]const u8, *const [N]u8, *const [N:S]u8,
/// []u8, [:S]u8, *[:S]u8, *[N:S]u8.
/// ```
/// These types are not considered strings:
/// ```
/// u8, [N]u8, [*]const u8, [*:0]const u8,
/// [*]const [N]u8, []const u16, []const i8,
/// *const u8, ?[]const u8, ?*const [N]u8.
/// ```
pub fn isZigString(comptime T: type) bool {
    return comptime blk: {
        // Only pointer types can be strings, no optionals
        const info = @typeInfo(T);
        if (info != .Pointer) break :blk false;
        const ptr = &info.Pointer;
        // Check for CV qualifiers that would prevent coerction to []const u8
        if (ptr.is_volatile or ptr.is_allowzero) break :blk false;
        // If it's already a slice, simple check.
        if (ptr.size == .Slice) {
            break :blk ptr.child == u8;
        }
        // Otherwise check if it's an array type that coerces to slice.
        if (ptr.size == .One) {
            const child = @typeInfo(ptr.child);
            if (child == .Array) {
                const arr = &child.Array;
                break :blk arr.child == u8;
            }
        }
        break :blk false;
    };
}
test "isZigString" {
    const testing = std.testing;
    try testing.expect(isZigString([]const u8));
    try testing.expect(isZigString([]u8));
    try testing.expect(isZigString([:0]const u8));
    try testing.expect(isZigString([:0]u8));
    try testing.expect(isZigString([:5]const u8));
    try testing.expect(isZigString([:5]u8));
    try testing.expect(isZigString(*const [0]u8));
    try testing.expect(isZigString(*[0]u8));
    try testing.expect(isZigString(*const [0:0]u8));
    try testing.expect(isZigString(*[0:0]u8));
    try testing.expect(isZigString(*const [0:5]u8));
    try testing.expect(isZigString(*[0:5]u8));
    try testing.expect(isZigString(*const [10]u8));
    try testing.expect(isZigString(*[10]u8));
    try testing.expect(isZigString(*const [10:0]u8));
    try testing.expect(isZigString(*[10:0]u8));
    try testing.expect(isZigString(*const [10:5]u8));
    try testing.expect(isZigString(*[10:5]u8));
    try testing.expect(!isZigString(u8));
    try testing.expect(!isZigString([4]u8));
    try testing.expect(!isZigString([4:0]u8));
    try testing.expect(!isZigString([*]const u8));
    try testing.expect(!isZigString([*]const [4]u8));
    try testing.expect(!isZigString([*c]const u8));
    try testing.expect(!isZigString([*c]const [4]u8));
    try testing.expect(!isZigString([*:0]const u8));
    try testing.expect(!isZigString([*:0]const u8));
    try testing.expect(!isZigString(*[]const u8));
    try testing.expect(!isZigString(?[]const u8));
    try testing.expect(!isZigString(?*const [4]u8));
    try testing.expect(!isZigString([]allowzero u8));
    try testing.expect(!isZigString([]volatile u8));
    try testing.expect(!isZigString(*allowzero [4]u8));
    try testing.expect(!isZigString(*volatile [4]u8));
}
