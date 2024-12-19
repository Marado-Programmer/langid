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
const set = @import("../../utils/set.zig");
const neural = @import("../../utils/neural.zig");
const Neuron = neural.Neuron;
const Layer = neural.Layer;
const Network = neural.Network;

const Params = struct { input: ?std.fs.File = null, vocab: ?std.fs.File = null };
const ParamsSpecificationError = error{ InvalidParamValue, ParamRepetition };

pub fn main(allocator: std.mem.Allocator, args: [][]const u8) !void {
    var params = Params{};
    defer if (params.input) |file| file.close();
    defer if (params.vocab) |file| file.close();

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

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var vocab = std.ArrayList([]const u8).init(allocator);
    defer vocab.deinit();

    while (try next(arena_allocator, params.vocab.?.reader())) |line| {
        if (line.len == 0) {
            continue;
        }
        try vocab.append(line);
    }

    const TrainingData = struct { class: []const u8, in: []f32, out: []f32 = undefined, raw: []const u8 };
    var phrases = std.ArrayList(TrainingData).init(allocator);
    defer phrases.deinit();
    var classes = set.Set([]const u8).init(allocator);
    defer classes.deinit();

    var map = std.StringHashMap(u8).init(allocator);
    defer map.deinit();

    while (try next(arena_allocator, params.input.?.reader())) |line| {
        if (line.len == 0) {
            continue;
        }
        map.clearRetainingCapacity();

        const separation = std.mem.indexOfAny(u8, line, " \n\r\t") orelse continue;
        const class = line[0..separation];
        try classes.put(class);
        var iterator = std.mem.splitAny(u8, line[separation..], " .,:;\"'\\/|_-{[()]}\n\r\t");
        while (iterator.next()) |word| {
            if (word.len == 0) {
                continue;
            }
            try map.put(word, (map.get(word) orelse 0) + 1);
        }

        const phrase = try arena_allocator.alloc(f32, vocab.items.len);
        for (vocab.items, 0..) |w, i| {
            phrase[i] = @as(f32, @floatFromInt(map.get(w) orelse 0));
        }

        try phrases.append(TrainingData{
            .class = class,
            .in = phrase,
            .raw = line[separation..],
        });
    }

    for (phrases.items, 0..) |v, i| {
        const out = try arena_allocator.alloc(f32, classes.map.count());
        var iterator = classes.map.iterator();
        var j: usize = 0;
        while (iterator.next()) |entry| {
            out[j] = if (std.mem.eql(u8, v.class, entry.key_ptr.*)) 1 else 0;
            j += 1;
        }
        phrases.items[i].out = out;
    }

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    var nn = try Network().init(allocator, vocab.items.len + 1, @constCast(&[3]usize{ vocab.items.len * 2, vocab.items.len, classes.count() }));
    defer nn.deinit(allocator);
    nn.randomize(rand);
    nn.learning_rate = 1e-2;
    for (nn.layers, 0..) |_, i| {
        nn.layers[i].set_activation_functions(.sigmoid);
    }

    var in = try mat.Matrix(f32, 2).init_alloc(allocator, [2]usize{ 1, vocab.items.len + 1 });
    defer in.deinit(allocator);
    try in.set([2]usize{ 0, 0 }, 1); // bias

    var i: usize = 0;
    var last_err: ?f32 = null;
    const tolerance = 1e-3;
    var stagnation_count: usize = 0;
    const max_stagnation_iterations = 5;
    const learning_amount: usize = 1e2;

    while (i < learning_amount) : (i += 1) {
        for (phrases.items) |phrase| {
            std.mem.copyForwards(f32, in.buf[1..], phrase.in);
            const out = try mat.Matrix(f32, 2).init(phrase.out, [2]usize{ phrase.out.len, 1 });
            std.log.debug("{d} {s}", .{ learning_amount - i, phrase.raw });
            _ = try nn.feed_forward(in, out);
        }

        const cur_err = nn.err();
        if (last_err != null and std.math.approxEqAbs(f32, cur_err, last_err.?, tolerance)) {
            //stagnation_count += 1;
            stagnation_count = 0;
            if (stagnation_count >= max_stagnation_iterations) {
                break;
            }
        } else {
            stagnation_count = 0;
        }

        last_err = cur_err;

        nn.use_training();
        std.log.debug("{d}", .{learning_amount - i});
    }

    nn.batch = false;

    const stdin_file = std.io.getStdIn().reader();
    var br = std.io.bufferedReader(stdin_file);
    const stdin = br.reader();
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    const out = try mat.Matrix(f32, 2).init_alloc(allocator, [2]usize{ classes.count(), 1 });
    defer out.deinit(allocator);

    try stdout.print("Available classes:\n", .{});
    var classes_iterator = classes.iterator();
    while (classes_iterator.next()) |class| {
        try stdout.print("\t- {s}\n", .{class.*});
    }
    try stdout.print("Write a phrase (CLASS phrase...): ", .{});
    try bw.flush();
    while (try stdin.readUntilDelimiterOrEofAlloc(arena_allocator, '\n', 1024)) |line| {
        if (line.len == 0) {
            break;
        }

        const separation = std.mem.indexOfAny(u8, line, " \n\r\t") orelse @panic("");
        const class = line[0..separation];

        if (classes.map.get(class) == null) {
            @panic("");
        }

        var iterator_classes = classes.map.iterator();
        var j: usize = 0;
        while (iterator_classes.next()) |entry| {
            out.buf[j] = if (std.mem.eql(u8, class, entry.key_ptr.*)) 1 else 0;
            j += 1;
        }

        map.clearRetainingCapacity();

        var iterator = std.mem.splitAny(u8, line[separation..], " .,:;\"'\\/|_-{[()]}\n\r\t");
        while (iterator.next()) |word| {
            if (word.len == 0) {
                continue;
            }
            try map.put(word, (map.get(word) orelse 0) + 1);
        }

        for (vocab.items, 1..) |w, k| {
            in.buf[k] = @as(f32, @floatFromInt(map.get(w) orelse 0));
        }

        const err = nn.err();

        const y = try nn.feed_forward(in, out);

        try stdout.print("y = {any}\n", .{y.buf});
        try stdout.print("error = {d}\n", .{err});
        try stdout.print("Write a phrase (CLASS phrase...): ", .{});
        try bw.flush();
    }

    try bw.flush();
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
