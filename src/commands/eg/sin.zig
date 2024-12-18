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
pub fn generate_data(data: *[][]f32, rand: std.Random) void {
    for (data.*, 0..) |_, n| {
        const x = random(rand);

        data.*[n][0] = x;
        data.*[n][1] = @sin(x);
    }
}

fn random(rand: std.Random) f32 {
    return std.math.degreesToRadians(@as(f32, @floatFromInt(rand.intRangeAtMost(i16, -180, 180))));
}
