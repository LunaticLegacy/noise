// Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)
// see LICENSE.txt for details
// $Id$

#include "Python.h"
#include <math.h>
#include <float.h>
#include "_noise.h"

// For numpy array support
#include "numpy/arrayobject.h"

// 2D simplex skew factors
#define F2 0.3660254037844386f  // 0.5 * (sqrt(3.0) - 1.0)
#define G2 0.21132486540518713f // (3.0 - sqrt(3.0)) / 6.0

float 
noise2(float x, float y) 
{
	int i1, j1, ii, jj, c;
	float s = (x + y) * F2;
	float i = floorf(x + s);
	float j = floorf(y + s);
	float t = (i + j) * G2;

	float xx[3], yy[3], f[3];
	float noise[3] = {0.0f, 0.0f, 0.0f};
	int g[3];

	xx[0] = x - (i - t);
	yy[0] = y - (j - t);

	i1 = (xx[0] > yy[0]) ? 1 : 0;
	j1 = (xx[0] <= yy[0]) ? 1 : 0;

	xx[2] = xx[0] + G2 * 2.0f - 1.0f;
	yy[2] = yy[0] + G2 * 2.0f - 1.0f;
	xx[1] = xx[0] - (float)i1 + G2;
	yy[1] = yy[0] - (float)j1 + G2;

	ii = ((int) i) & 255;
	jj = ((int) j) & 255;
	g[0] = PERM[ii + PERM[jj]] % 12;
	g[1] = PERM[ii + i1 + PERM[jj + j1]] % 12;
	g[2] = PERM[ii + 1 + PERM[jj + 1]] % 12;

	for (c = 0; c <= 2; c++)
		f[c] = 0.5f - xx[c]*xx[c] - yy[c]*yy[c];
	
	for (c = 0; c <= 2; c++)
		if (f[c] > 0)
			noise[c] = f[c]*f[c]*f[c]*f[c] * (GRAD3[g[c]][0]*xx[c] + GRAD3[g[c]][1]*yy[c]);
	
	return (noise[0] + noise[1] + noise[2]) * 70.0f;
}

#define dot3(v1, v2) ((v1)[0]*(v2)[0] + (v1)[1]*(v2)[1] + (v1)[2]*(v2)[2])

#define ASSIGN(a, v0, v1, v2) {(a)[0] = (float)(v0); (a)[1] = (float)(v1); (a)[2] = (float)(v2);}

#define F3 (1.0f / 3.0f)
#define G3 (1.0f / 6.0f)

float 
noise3(float x, float y, float z) 
{
	int c, o1[3], o2[3], g[4], ii, jj, kk;
	float f[4], noise[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float s = (x + y + z) * F3;
	float i = floorf(x + s);
	float j = floorf(y + s);
	float k = floorf(z + s);
	float t = (i + j + k) * G3;

	float pos[4][3];

	pos[0][0] = x - (i - t);
	pos[0][1] = y - (j - t);
	pos[0][2] = z - (k - t);

	if (pos[0][0] >= pos[0][1]) {
		if (pos[0][1] >= pos[0][2]) {
			ASSIGN(o1, 1, 0, 0);
			ASSIGN(o2, 1, 1, 0);
		} else if (pos[0][0] >= pos[0][2]) {
			ASSIGN(o1, 1, 0, 0);
			ASSIGN(o2, 1, 0, 1);
		} else {
			ASSIGN(o1, 0, 0, 1);
			ASSIGN(o2, 1, 0, 1);
		}
	} else {
		if (pos[0][1] < pos[0][2]) {
			ASSIGN(o1, 0, 0, 1);
			ASSIGN(o2, 0, 1, 1);
		} else if (pos[0][0] < pos[0][2]) {
			ASSIGN(o1, 0, 1, 0);
			ASSIGN(o2, 0, 1, 1);
		} else {
			ASSIGN(o1, 0, 1, 0);
			ASSIGN(o2, 1, 1, 0);
		}
	}
	
	for (c = 0; c <= 2; c++) {
		pos[3][c] = pos[0][c] - 1.0f + 3.0f * G3;
		pos[2][c] = pos[0][c] - (float)o2[c] + 2.0f * G3;
		pos[1][c] = pos[0][c] - (float)o1[c] + G3;
	}

	ii = ((int) i) & 255; 
	jj = ((int) j) & 255; 
	kk = ((int) k) & 255;
	g[0] = PERM[ii + PERM[jj + PERM[kk]]] % 12;
	g[1] = PERM[ii + o1[0] + PERM[jj + o1[1] + PERM[o1[2] + kk]]] % 12;
	g[2] = PERM[ii + o2[0] + PERM[jj + o2[1] + PERM[o2[2] + kk]]] % 12;
	g[3] = PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1]]] % 12; 

	for (c = 0; c <= 3; c++) {
		f[c] = 0.6f - pos[c][0]*pos[c][0] - pos[c][1]*pos[c][1] - pos[c][2]*pos[c][2];
	}
	
	for (c = 0; c <= 3; c++) {
		if (f[c] > 0) {
			noise[c] = f[c]*f[c]*f[c]*f[c] * dot3(pos[c], GRAD3[g[c]]);
		}
	}
	
	return (noise[0] + noise[1] + noise[2] + noise[3]) * 32.0f;
}

static inline float
fbm_noise3(float x, float y, float z, int octaves, float persistence, float lacunarity) {
    float freq = 1.0f;
    float amp = 1.0f;
    float max = 1.0f;
    float total = noise3(x, y, z);
    int i;

    for (i = 1; i < octaves; ++i) {
        freq *= lacunarity;
        amp *= persistence;
        max += amp;
        total += noise3(x * freq, y * freq, z * freq) * amp;
    }
    return total / max;
}

#define dot4(v1, x, y, z, w) ((v1)[0]*(x) + (v1)[1]*(y) + (v1)[2]*(z) + (v1)[3]*(w))

#define F4 0.30901699437494745f /* (sqrt(5.0) - 1.0) / 4.0 */
#define G4 0.1381966011250105f /* (5.0 - sqrt(5.0)) / 20.0 */

float 
noise4(float x, float y, float z, float w) {
    float noise[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float s = (x + y + z + w) * F4;
    float i = floorf(x + s);
    float j = floorf(y + s);
    float k = floorf(z + s);
    float l = floorf(w + s);
    float t = (i + j + k + l) * G4;

    float x0 = x - (i - t);
    float y0 = y - (j - t);
    float z0 = z - (k - t);
    float w0 = w - (l - t);

    int c = ((x0 > y0)*32 + (x0 > z0)*16 + (y0 > z0)*8 + (x0 > w0)*4 + (y0 > w0)*2 + (z0 > w0));
    int i1 = (SIMPLEX[c][0]>=3) ? 1 : 0;
    int j1 = (SIMPLEX[c][1]>=3) ? 1 : 0;
    int k1 = (SIMPLEX[c][2]>=3) ? 1 : 0;
    int l1 = (SIMPLEX[c][3]>=3) ? 1 : 0;
    int i2 = (SIMPLEX[c][0]>=2) ? 1 : 0;
    int j2 = (SIMPLEX[c][1]>=2) ? 1 : 0;
    int k2 = (SIMPLEX[c][2]>=2) ? 1 : 0;
    int l2 = (SIMPLEX[c][3]>=2) ? 1 : 0;
    int i3 = (SIMPLEX[c][0]>=1) ? 1 : 0;
    int j3 = (SIMPLEX[c][1]>=1) ? 1 : 0;
    int k3 = (SIMPLEX[c][2]>=1) ? 1 : 0;
    int l3 = (SIMPLEX[c][3]>=1) ? 1 : 0;

    float x1 = x0 - (float)i1 + G4;
    float y1 = y0 - (float)j1 + G4;
    float z1 = z0 - (float)k1 + G4;
    float w1 = w0 - (float)l1 + G4;
    float x2 = x0 - (float)i2 + 2.0f*G4;
    float y2 = y0 - (float)j2 + 2.0f*G4;
    float z2 = z0 - (float)k2 + 2.0f*G4;
    float w2 = w0 - (float)l2 + 2.0f*G4;
    float x3 = x0 - (float)i3 + 3.0f*G4;
    float y3 = y0 - (float)j3 + 3.0f*G4;
    float z3 = z0 - (float)k3 + 3.0f*G4;
    float w3 = w0 - (float)l3 + 3.0f*G4;
    float x4 = x0 - 1.0f + 4.0f*G4;
    float y4 = y0 - 1.0f + 4.0f*G4;
    float z4 = z0 - 1.0f + 4.0f*G4;
    float w4 = w0 - 1.0f + 4.0f*G4;

    int ii = ((int)i) & 255;
    int jj = ((int)j) & 255;
    int kk = ((int)k) & 255;
    int ll = ((int)l) & 255;
    int gi0 = PERM[ii + PERM[jj + PERM[kk + PERM[ll]]]] & 0x1f;
    int gi1 = PERM[ii + i1 + PERM[jj + j1 + PERM[kk + k1 + PERM[ll + l1]]]] & 0x1f; 
    int gi2 = PERM[ii + i2 + PERM[jj + j2 + PERM[kk + k2 + PERM[ll + l2]]]] & 0x1f; 
    int gi3 = PERM[ii + i3 + PERM[jj + j3 + PERM[kk + k3 + PERM[ll + l3]]]] & 0x1f; 
    int gi4 = PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1 + PERM[ll + 1]]]] & 0x1f;
    float t0, t1, t2, t3, t4;

    t0 = 0.6f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
    if (t0 >= 0.0f) {
        t0 *= t0;
        noise[0] = t0 * t0 * dot4(GRAD4[gi0], x0, y0, z0, w0);
    }
    t1 = 0.6f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    if (t1 >= 0.0f) {
        t1 *= t1;
        noise[1] = t1 * t1 * dot4(GRAD4[gi1], x1, y1, z1, w1);
    }
    t2 = 0.6f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    if (t2 >= 0.0f) {
        t2 *= t2;
        noise[2] = t2 * t2 * dot4(GRAD4[gi2], x2, y2, z2, w2);
    }
    t3 = 0.6f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    if (t3 >= 0.0f) {
        t3 *= t3;
        noise[3] = t3 * t3 * dot4(GRAD4[gi3], x3, y3, z3, w3);
    }
    t4 = 0.6f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    if (t4 >= 0.0f) {
        t4 *= t4;
        noise[4] = t4 * t4 * dot4(GRAD4[gi4], x4, y4, z4, w4);
    }

    return 27.0f * (noise[0] + noise[1] + noise[2] + noise[3] + noise[4]);
}

static inline float
fbm_noise4(float x, float y, float z, float w, int octaves, float persistence, float lacunarity) {
    float freq = 1.0f;
    float amp = 1.0f;
    float max = 1.0f;
    float total = noise4(x, y, z, w);
    int i;

    for (i = 1; i < octaves; ++i) {
        freq *= lacunarity;
        amp *= persistence;
        max += amp;
        total += noise4(x * freq, y * freq, z * freq, w * freq) * amp;
    }
    return total / max;
}


static PyObject *
py_noise2(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x, y;
	int octaves = 1;
	float persistence = 0.5f;
    float lacunarity = 2.0f;
    float repeatx = FLT_MAX;
    float repeaty = FLT_MAX;
    float z = 0.0f;
	static char *kwlist[] = {"x", "y", "octaves", "persistence", "lacunarity", 
        "repeatx", "repeaty", "base", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ff|ifffff:snoise2", kwlist,
		&x, &y, &octaves, &persistence, &lacunarity, &repeatx, &repeaty, &z)) {
		return NULL;
    }
    if (octaves <= 0) {
        PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
        return NULL;
    }
	
    if (repeatx == FLT_MAX && repeaty == FLT_MAX) {
        // Flat noise, no tiling
        float freq = 1.0f;
        float amp = 1.0f;
        float max = 1.0f;
        float total = noise2(x + z, y + z);
        int i;

        for (i = 1; i < octaves; i++) {
            freq *= lacunarity;
            amp *= persistence;
            max += amp;
            total += noise2(x * freq + z, y * freq + z) * amp;
        }
        return (PyObject *) PyFloat_FromDouble((double) (total / max));
    } else { // Tiled noise
        float w = z;
        if (repeaty != FLT_MAX) {
            float yf = y * 2.0f / repeaty;
            float yr = repeaty * (float)M_1_PI * 0.5f;
            float vy = fast_sin(yf);
            float vyz = fast_cos(yf);
            y = vy * yr;
            w += vyz * yr;
            if (repeatx == FLT_MAX) {
                return (PyObject *) PyFloat_FromDouble(
                    (double) fbm_noise3(x, y, w, octaves, persistence, lacunarity));
            }
        }
        if (repeatx != FLT_MAX) {
            float xf = x * 2.0f / repeatx;
            float xr = repeatx * (float)M_1_PI * 0.5f;
            float vx = fast_sin(xf);
            float vxz = fast_cos(xf);
            x = vx * xr;
            z += vxz * xr;
            if (repeaty == FLT_MAX) {
                return (PyObject *) PyFloat_FromDouble(
                    (double) fbm_noise3(x, y, z, octaves, persistence, lacunarity));
            }
        }
        return (PyObject *) PyFloat_FromDouble(
            (double) fbm_noise4(x, y, z, w, octaves, persistence, lacunarity));
    }
}

static PyObject *
py_noise3(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x, y, z;
	int octaves = 1;
	float persistence = 0.5f;
    float lacunarity = 2.0f;

	static char *kwlist[] = {"x", "y", "z", "octaves", "persistence", "lacunarity", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff|iff:snoise3", kwlist,
		&x, &y, &z, &octaves, &persistence, &lacunarity))
		return NULL;
	
	if (octaves == 1) {
		// Single octave, return simple noise
		return (PyObject *) PyFloat_FromDouble((double) noise3(x, y, z));
	} else if (octaves > 1) {
		return (PyObject *) PyFloat_FromDouble(
            (double) fbm_noise3(x, y, z, octaves, persistence, lacunarity));
	} else {
		PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
		return NULL;
	}
}

static PyObject *
py_noise4(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x, y, z, w;
	int octaves = 1;
	float persistence = 0.5f;
	float lacunarity = 2.0f;

	static char *kwlist[] = {"x", "y", "z", "w", "octaves", "persistence", "lacunarity", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ffff|iff:snoise4", kwlist,
		&x, &y, &z, &w, &octaves, &persistence, &lacunarity))
		return NULL;
	
	if (octaves == 1) {
		// Single octave, return simple noise
		return (PyObject *) PyFloat_FromDouble((double) noise4(x, y, z, w));
	} else if (octaves > 1) {
		return (PyObject *) PyFloat_FromDouble(
            (double) fbm_noise4(x, y, z, w, octaves, persistence, lacunarity));
	} else {
		PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
		return NULL;
	}
}

/**
 * def batch_snoise2(
 *     min_x: float, min_y: float,
 *     max_x: float, max_y: float,
 * 	   repeat_x: float, repeat_y: float,
 *     base: float, resolution: float,
 *     callback: Optional[Callable] = None
 * ) -> np.ndarray[np.float32]
 * 
 */
static PyObject* py_batch_snoise2(PyObject* self, PyObject* args, PyObject* kwargs) {

    // Prepare parameters.
    float min_x, min_y, max_x, max_y;
	float repeat_x = FLT_MAX, repeat_y = FLT_MAX;
	float base = 0.0f;
	float resolution = 30.0f;  // 30 units.
	PyObject* callback = NULL;
    
    static char* kwlist[] = {
        "min_x", "min_y",
        "max_x", "max_y",
		"repeat_x", "repeat_y",
		"base", "resolution", 
		"callback",
		NULL
    };
    
    // Parse parameter.
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ffff|ffffO:batch_noise2", kwlist,
        &min_x, &min_y, 
		&max_x, &max_y, 
		&repeat_x, &repeat_y, 
		&base, &resolution,
		&callback)
	) {
        return NULL;
    }

	// Validate the callback function.
    if (callback && callback != Py_None && !PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }

    // Calculate grid dimensions
    int width = (int)((max_x - min_x) / resolution) + 1;
    int height = (int)((max_y - min_y) / resolution) + 1;
    
    if (width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid grid dimensions. I meant, min should be smaller than max.");
        return NULL;
    }

	// Create a numpy ndarray.
	npy_intp dims[2] = {height, width};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!result_array) {
		PySys_WriteStdout("WARNING: No array created.\n");
		return NULL;
	}
    float* data = (float*)PyArray_DATA(result_array);

    float step_x = (max_x - min_x) / (float)(width - 1);
    float step_y = (max_y - min_y) / (float)(height - 1);

	// Iter.
	for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float x = min_x + i * step_x;
            float y = min_y + j * step_y;
            
            // Use the same logic as in py_noise2 for tiled vs non-tiled noise
            float val;
            if (repeat_x == FLT_MAX && repeat_y == FLT_MAX) {
                // Flat noise, no tiling
                val = noise2(x, y);
            } else {
                // Tiled noise - use the same approach as py_noise2
                float z = base;
                float w = base;
                if (repeat_y != FLT_MAX) {
                    float yf = y * 2.0f / repeat_y;
                    float yr = repeat_y * (float)M_1_PI * 0.5f;
                    float vy = fast_sin(yf);
                    float vyz = fast_cos(yf);
                    y = vy * yr;
                    w += vyz * yr;
                }
                if (repeat_x != FLT_MAX) {
                    float xf = x * 2.0f / repeat_x;
                    float xr = repeat_x * (float)M_1_PI * 0.5f;
                    float vx = fast_sin(xf);
                    float vxz = fast_cos(xf);
                    x = vx * xr;
                    z += vxz * xr;
                }
                
                if (repeat_x == FLT_MAX || repeat_y == FLT_MAX) {
                    val = fbm_noise3(x, y, z, 1, 0.5f, 2.0f);
                } else {
                    val = fbm_noise4(x, y, z, w, 1, 0.5f, 2.0f);
                }
            }
            
            data[j * width + i] = val;
        }

		// Callback, once per row.
		if (callback && callback != Py_None) {
            double progress = (double)(j + 1) / (double)height;  // in range [0, 1]
            PyObject* arg = Py_BuildValue("(d)", progress);      // tuple with one float
			if (!arg) {
                Py_DECREF(result_array);
				PySys_WriteStderr("Error in building callback arg.\n");
                return NULL;
            }
            PyObject* res = PyObject_CallObject(callback, arg);
            Py_DECREF(arg);

            if (!res) {  // exception occurred in Python
                Py_DECREF(result_array);
				PySys_WriteStderr("Error in calling callback func.\n");
                return NULL;
            }
            Py_DECREF(res);
        }
    }

    return (PyObject*)result_array;
}

/**
 * def batch_snoise3(
 *     min_x: float, min_y: float, min_z: float,
 *     max_x: float, max_y: float, max_z: float,
 * 	   repeat_x: float, repeat_y: float, repeat_z: float,
 *     base: float, resolution: float,
 *     callback: Optional[Callable] = None
 * ) -> np.ndarray[np.float32]
 * 
 */
static PyObject* py_batch_snoise3(PyObject* self, PyObject* args, PyObject* kwargs) {

    // Prepare parameters.
    float min_x, min_y, min_z, max_x, max_y, max_z;
	float repeat_x = 1024.0f, repeat_y = 1024.0f, repeat_z = 1024.0f;
	float base = 0.0f;
	float resolution = 30.0f;  // 30 units.
	PyObject* callback = NULL;
    
    static char* kwlist[] = {
        "min_x", "min_y", "min_z",
        "max_x", "max_y", "max_z",
		"repeat_x", "repeat_y", "repeat_z",
		"base", "resolution", 
		"callback",
		NULL
    };
    
    // Parse parameter.
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ffffff|fffffO:batch_noise3", kwlist,
        &min_x, &min_y, &min_z,
		&max_x, &max_y, &max_z,
		&repeat_x, &repeat_y, &repeat_z,
		&base, &resolution,
		&callback)
	) {
        return NULL;
    }

	// Validate the callback function.
    if (callback && callback != Py_None && !PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }

    // Calculate grid dimensions
    int width = (int)((max_x - min_x) / resolution) + 1;
    int height = (int)((max_y - min_y) / resolution) + 1;
	int depth = (int)((max_z - min_z) / resolution) + 1;
    
    if (width <= 0 || height <= 0 || depth <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid grid dimensions. I meant, min should be smaller than max.");
        return NULL;
    }

	// Create a numpy ndarray.
	npy_intp dims[3] = {depth, height, width};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result_array) {
		PySys_WriteStdout("WARNING: No array created.\n");
		return NULL;
	}
    float* data = (float*)PyArray_DATA(result_array);

    float step_x = (max_x - min_x) / (float)(width - 1);
    float step_y = (max_y - min_y) / (float)(height - 1);
	float step_z = (max_z - min_z) / (float)(depth - 1);

	// Iter.
	for (int k = 0; k < depth; k++) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float x = min_x + i * step_x;
				float y = min_y + j * step_y;
				float z = min_z + k * step_z;
				float val = noise3(x, y, z);
				data[k * height * width + j * width + i] = val;
			}
		}

		// Callback, once per layer.
		if (callback && callback != Py_None) {
			double progress = (double)(k + 1) / (double)depth;  // in range [0, 1]
			PyObject* arg = Py_BuildValue("(d)", progress);      // tuple with one float
			if (!arg) {
				Py_DECREF(result_array);
				PySys_WriteStderr("Error in building callback arg.\n");
				return NULL;
			}
			PyObject* res = PyObject_CallObject(callback, arg);
			Py_DECREF(arg);

			if (!res) {  // exception occurred in Python
				Py_DECREF(result_array);
				PySys_WriteStderr("Error in calling callback func.\n");
				return NULL;
			}
			Py_DECREF(res);
		}
	}

    return (PyObject*)result_array;
}


static PyMethodDef simplex_functions[] = {
	{"noise2", (PyCFunction)py_noise2, METH_VARARGS | METH_KEYWORDS, 
		"noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=None, repeaty=None, base=0.0) "
        "return simplex noise value for specified 2D coordinate.\n\n"
		"octaves -- specifies the number of passes, defaults to 1 (simple noise).\n\n"
		"persistence -- specifies the amplitude of each successive octave relative\n"
		"to the one below it. Defaults to 0.5 (each higher octave's amplitude\n"
		"is halved). Note the amplitude of the first pass is always 1.0.\n\n"
        "lacunarity -- specifies the frequency of each successive octave relative\n"
        "to the one below it, similar to persistence. Defaults to 2.0.\n\n"
        "repeatx, repeaty -- specifies the interval along each axis when \n"
		"the noise values repeat. This can be used as the tile size for creating \n"
		"tileable textures\n\n"
		"base -- specifies a fixed offset for the noise coordinates. Useful for\n"
		"generating different noise textures with the same repeat interval"},
	{"noise3", (PyCFunction)py_noise3, METH_VARARGS | METH_KEYWORDS, 
		"noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for "
		"specified 3D coordinate\n\n"
		"octaves -- specifies the number of passes, defaults to 1 (simple noise).\n\n"
		"persistence -- specifies the amplitude of each successive octave relative\n"
		"to the one below it. Defaults to 0.5 (each higher octave's amplitude\n"
		"is halved). Note the amplitude of the first pass is always 1.0.\n\n"
        "lacunarity -- specifies the frequency of each successive octave relative\n"
        "to the one below it, similar to persistence. Defaults to 2.0."},
	{"noise4", (PyCFunction)py_noise4, METH_VARARGS | METH_KEYWORDS, 
		"noise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for "
		"specified 4D coordinate\n\n"
		"octaves -- specifies the number of passes, defaults to 1 (simple noise).\n\n"
		"persistence -- specifies the amplitude of each successive octave relative\n"
		"to the one below it. Defaults to 0.5 (each higher octave's amplitude\n"
		"is halved). Note the amplitude of the first pass is always 1.0.\n\n"
        "lacunarity -- specifies the frequency of each successive octave relative\n"
        "to the one below it, similar to persistence. Defaults to 2.0."},
	{"batch_noise2", (PyCFunction)py_batch_snoise2, METH_VARARGS | METH_KEYWORDS, 
		"batch_snoise2(\n"
		"min_x: float, min_y: float, max_x: float, max_y: float, "
		"repeat_x: float = None, repeat_y: float = None, base: float = 0.0, \n"
		"resolution: float = 30.0, callback: Optional[Callable] = None\n"
		")\n\n"
		"Generate a 2D array of Simplex noise values.\n\n"
		"min_x, min_y -- minimum coordinate values.\n"
		"max_x, max_y -- maximum coordinate values.\n"
		"repeat_x, repeat_y, base -- (see noise2 for more info)\n"
		"resolution -- number of samples per unit.\n"
		"callback -- optional callback function for acquiring the progress of noise generating."
	},
	{"batch_noise3", (PyCFunction)py_batch_snoise3, METH_VARARGS | METH_KEYWORDS,
		"batch_snoise3(\n"
		"min_x: float, min_y: float, min_z: float,\n"
        "max_x: float, max_y: float, max_z: float,\n"
        "repeat_x: float = 1024.0, repeat_y: float = 1024.0, repeat_z: float = 1024.0,\n"
        "base: float = 0.0,\n"
        "resolution: float = 30.0,\n"
        "callback: Optional[Callable] = None\n"
		")\n\n"
		"Generate a 3D array of Simplex noise values.\n\n"
		"min_x, min_y, min_z -- minimum coordinate values.\n"
		"max_x, max_y, max_z -- maximum coordinate values.\n"
		"repeat_x, repeat_y, repeat_z, base -- (see noise3 for more info)\n"
		"resolution -- number of samples per unit.\n"
		"callback -- optional callback function for acquiring the progress of noise generating."
	},
	{NULL}
};

PyDoc_STRVAR(module_doc, "Native-code simplex noise functions");

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"_simplex",
	module_doc,
	-1,                 /* m_size */
	simplex_functions,  /* m_methods */
	NULL,               /* m_reload (unused) */
	NULL,               /* m_traverse */
	NULL,               /* m_clear */
	NULL                /* m_free */
};

PyObject *
PyInit__simplex(void)
{
	import_array();
    return PyModule_Create(&moduledef);
}

#else

void
init_simplex(void)
{
	PyErr_SetString(PyExc_SystemError, "Version for Python 2.0 in this lib is DEPRECATED. Please upgrade your Python to Python3.");
}

#endif