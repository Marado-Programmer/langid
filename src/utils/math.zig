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
const math = @import("std").math;
pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + math.exp(-x));
}
pub fn sigmoidp(x: f32) f32 {
    return sigmoid(x) * (1 - sigmoid(x));
}

pub fn identity(x: f32) f32 {
    return x;
}
pub fn identityp(_: f32) f32 {
    return 1;
}

pub fn reLU(x: f32) f32 {
    return @max(x, 0);
}

pub fn reLUp(x: f32) f32 {
    return if (x > 0) 1 else 0;
}

pub fn lReLU(x: f32, a: f32) f32 {
    return if (x > 0) x else a * x;
}

pub fn lReLUp(x: f32, a: f32) f32 {
    return if (x > 0) 1 else a;
}

pub fn rosenblatt(x: f32) f32 {
    return if (x >= 0) 1 else -1;
}

pub fn rosenblattp(_: f32) f32 {
    return 0;
}

// soft max
// comulative distribution function

pub fn tanh(x: f32) f32 {
    return math.tanh(x);
}

pub fn tanhp(x: f32) f32 {
    return 1 - math.pow(f32, math.tanh(x), 2);
}
