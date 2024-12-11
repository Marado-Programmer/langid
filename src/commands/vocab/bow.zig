// LANGuage IDentification - To identify languages.
// Copyright (C) 2024  João Augusto Costa Branco Marado Torres
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
// for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
const std = @import("std");
const set = @import("../../utils/set.zig");

pub fn create_vocab(allocator: std.mem.Allocator, file: std.fs.File, sort: bool) !void {
    var bag = set.Set([]const u8).init(allocator);
    defer bag.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();
    while (try next(arena_allocator, file.reader())) |word| {
        var iterator = std.mem.splitAny(u8, word, ".,:;\"'\\/|_-{[()]}\n\r\t");
        while (iterator.next()) |w| {
            if (w.len > 0) {
                try bag.put(w);
            }
        }
    }

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    var iterator = bag.iterator();
    if (sort) {
        var sorted = try allocator.alloc([]const u8, bag.count());
        defer allocator.free(sorted);
        var i: u32 = 0;
        while (iterator.next()) |w| {
            sorted[i] = w.*;
            i += 1;
        }
        std.mem.sort([]const u8, sorted, {}, lessThan);
        for (sorted) |w| {
            try stdout.print("{s}\n", .{w});
        }
    } else {
        while (iterator.next()) |w| {
            try stdout.print("{s}\n", .{w.*});
        }
    }

    try bw.flush(); // don't forget to flush!
}

// https://zig.guide/standard-library/readers-and-writers
fn next(allocator: std.mem.Allocator, reader: anytype) !?[]const u8 {
    const line = (try reader.readUntilDelimiterOrEofAlloc(allocator, ' ', 1024)) orelse return null;
    // trim annoying windows-only carriage return character
    if (@import("builtin").os.tag == .windows) {
        return std.mem.trimRight(u8, line, "\r");
    } else {
        return line;
    }
}

// https://stackoverflow.com/questions/79012210/sorting-array-of-strings-alphabetically-in-ascendending-order-in-zig
fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}
