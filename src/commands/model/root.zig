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
const mat = @import("../../utils/matrix.zig");
const neural = @import("../../utils/neural.zig");
const Neuron = neural.Neuron;
const Layer = neural.Layer;
const Network = neural.Network;

const Params = struct { input: ?std.fs.File = null, vocab: ?std.fs.File = null };
const ParamsSpecificationError = error{ InvalidParamValue, ParamRepetition };

pub fn main(allocator: std.mem.Allocator, args: [][]const u8) !void {
    var params = Params{};
    defer if (params.input) |in| in.close();

    var skip = false;
    for (args, 0..) |arg, i| {
        if (skip) {
            skip = false;
            continue;
        }

        if (std.mem.eql(u8, arg, "--input")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const path_arg = args[i + 1];
            try specify_input(&params, path_arg);
        } else if (std.mem.startsWith(u8, arg, "--input=")) {
            const a = "--input=";
            try specify_input(&params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--vocab")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const path_arg = args[i + 1];
            try specify_vocab(&params, path_arg);
        } else if (std.mem.startsWith(u8, arg, "--vocab=")) {
            const a = "--vocab=";
            try specify_vocab(&params, arg[a.len..]);
        }
    }

    if (params.vocab == null) {
        @panic("NO VOCAB PROVIDED");
    }

    if (params.input == null) {
        params.input = std.io.getStdIn();
    }

    var vocab = std.ArrayList([]const u8).init(allocator);
    defer vocab.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();
    while (try next(arena_allocator, params.vocab.?.reader())) |line| {
        if (line.len == 0) {
            continue;
        }
        try vocab.append(line);
    }

    var neuron = try Neuron().init_alloc(allocator, vocab.items.len + 1);
    defer neuron.deinit(allocator);

    var layer = try Layer().init(allocator, 4, vocab.items.len + 1);
    defer layer.deinit(allocator);

    var nn = try Network(3).init(allocator, vocab.items.len + 1, [3]usize{ 4, 10, 3 });
    defer nn.deinit(allocator);

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    neuron.randomize(rand);
    neuron.set_f(false);

    layer.randomize(rand);
    neuron.set_f(false);

    nn.randomize(rand);

    var phrase = try mat.Matrix(f32, 2).init_alloc(allocator, [2]usize{ 1, vocab.items.len + 1 });
    defer phrase.deinit(allocator);

    try phrase.set([2]usize{ 0, 0 }, 1); // bias

    var map = std.StringHashMap(u8).init(allocator);
    defer map.deinit();

    while (try next(arena_allocator, params.input.?.reader())) |line| {
        if (line.len == 0) {
            continue;
        }
        map.clearRetainingCapacity();

        var iterator = std.mem.splitAny(u8, line, " .,:;\"'\\/|_-{[()]}\n\r\t");
        while (iterator.next()) |word| {
            try map.put(word, (map.get(word) orelse 0) + 1);
        }

        for (vocab.items, 1..) |w, i| {
            try phrase.set([2]usize{ 0, i }, @as(f32, @floatFromInt(map.get(w) orelse 0)));
        }

        try neuron.calc_activation(phrase);
        std.log.debug("{d}\t`{s}`", .{ neuron.get_activation(), line });
        try layer.calc_activation(phrase);
        std.log.debug("{any}\t`{s}`", .{ layer.activations.buf, line });
        const y = try nn.feed_forward(phrase);
        std.log.debug("{any}\t`{s}`", .{ y.buf, line });
    }
}

// https://zig.guide/standard-library/readers-and-writers
fn next(allocator: std.mem.Allocator, reader: anytype) !?[]const u8 {
    const line = (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 1024)) orelse return null;
    // trim annoying windows-only carriage return character
    if (@import("builtin").os.tag == .windows) {
        return std.mem.trimRight(u8, line, "\r");
    } else {
        return line;
    }
}

fn specify_input(params: *Params, path_arg: []const u8) (std.posix.RealPathError || std.fs.File.OpenError || ParamsSpecificationError)!void {
    var path_buffer: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fs.realpath(path_arg, &path_buffer);

    if (params.input) |in| in.close();

    params.input = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
}

fn specify_vocab(params: *Params, path_arg: []const u8) (std.posix.RealPathError || std.fs.File.OpenError || ParamsSpecificationError)!void {
    var path_buffer: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fs.realpath(path_arg, &path_buffer);

    if (params.vocab) |in| in.close();

    params.vocab = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
}
