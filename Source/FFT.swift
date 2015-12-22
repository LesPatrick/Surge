// FFT.swift
//
// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Accelerate

// MARK: Fast Fourier Transform

public func fft(input: [Float]) -> [Float] {
    var real = [Float](input)
    var imaginary = [Float](count: input.count, repeatedValue: 0.0)
    var splitComplex = DSPSplitComplex(realp: &real, imagp: &imaginary)

    let length = vDSP_Length(floor(log2(Float(input.count))))
    let radix = FFTRadix(kFFTRadix2)
    let weights = vDSP_create_fftsetup(length, radix)
    vDSP_fft_zip(weights, &splitComplex, 1, length, FFTDirection(FFT_FORWARD))

    var magnitudes = [Float](count: input.count, repeatedValue: 0.0)
    vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(input.count))

    var normalizedMagnitudes = [Float](count: input.count, repeatedValue: 0.0)
    vDSP_vsmul(sqrt(magnitudes), 1, [2.0 / Float(input.count)], &normalizedMagnitudes, 1, vDSP_Length(input.count))

    vDSP_destroy_fftsetup(weights)

    return normalizedMagnitudes
}

public func fft(input: [Double]) -> [Double] {
    var real = [Double](input)
    var imaginary = [Double](count: input.count, repeatedValue: 0.0)
    var splitComplex = DSPDoubleSplitComplex(realp: &real, imagp: &imaginary)

    let length = vDSP_Length(floor(log2(Float(input.count))))
    let radix = FFTRadix(kFFTRadix2)
    let weights = vDSP_create_fftsetupD(length, radix)
    vDSP_fft_zipD(weights, &splitComplex, 1, length, FFTDirection(FFT_FORWARD))

    var magnitudes = [Double](count: input.count, repeatedValue: 0.0)
    vDSP_zvmagsD(&splitComplex, 1, &magnitudes, 1, vDSP_Length(input.count))

    var normalizedMagnitudes = [Double](count: input.count, repeatedValue: 0.0)
    vDSP_vsmulD(sqrt(magnitudes), 1, [2.0 / Double(input.count)], &normalizedMagnitudes, 1, vDSP_Length(input.count))

    vDSP_destroy_fftsetupD(weights)

    return normalizedMagnitudes
}

// MARK: Discrete Fourier Transform

public func dft(input: [Double]) -> (real:[Double], imag: [Double], magnitudes: [Double]) {
    let log2n = vDSP_Length(floor(log2(Double(input.count))))
    var realIn = [Double](count: input.count/2, repeatedValue: 0.0)
    var imagIn = [Double](count: input.count/2, repeatedValue: 0.0)
    var realOut = [Double](count: input.count, repeatedValue: 0.0)
    var imagOut = [Double](count: input.count, repeatedValue: 0.0)
    var magnitudes = [Double](count: input.count, repeatedValue: 0.0)
    var inComplex = DSPDoubleSplitComplex(realp: &realIn, imagp: &imagIn)
    var splitComplex = DSPDoubleSplitComplex(realp: &realOut, imagp: &imagOut)
    
    for (var i = 0; i < input.count; i+=2) {
        inComplex.realp[i/2] = input[i]
        inComplex.imagp[i/2] = input[i+1]
    }
    
    let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2))
    
    vDSP_fft_zripD(setup, &inComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))
    
    for (var i = 0; i < input.count/2; i++) {
        inComplex.realp[i] *= 0.5
        inComplex.imagp[i] *= 0.5
    }
    
    splitComplex.realp[0] = inComplex.realp[0]
    for (var i = 1; i < input.count/2; i++) {
        splitComplex.realp[i] = inComplex.realp[i]
        splitComplex.realp[input.count - i] = inComplex.realp[i]
        splitComplex.imagp[i] = inComplex.imagp[i]
        splitComplex.imagp[input.count - i] = inComplex.imagp[i]
    }
    splitComplex.realp[input.count/2] = inComplex.imagp[0]
    
    magnitudes = sqrt(add(mul(realOut, realOut), mul(imagOut, imagOut)))
    
    return (real: realOut, imag: imagOut, magnitudes: magnitudes)
}

