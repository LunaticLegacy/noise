// Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)
// see LICENSE.txt for details
// $Id$

#include "Python.h"
#include <math.h>
#include <stdio.h>
#include "_noise.h"

#include "numpy/arrayobject.h"

#ifdef _MSC_VER
#define inline __inline
#endif

#define lerp(t, a, b) ((a) + (t) * ((b) - (a)))

static inline float
grad1(const int hash, const float x)
{
	float g = (hash & 7) + 1.0f;
	if (hash & 8)
		g = -1;
	return (g * x);
}

float
noise1(float x, const int repeat, const int base)
{
	float fx;
	int i = (int)floorf(x) % repeat;
	int ii = (i + 1) % repeat;
	i = (i & 255) + base;
	ii = (ii & 255) + base;

	x -= floorf(x);
	fx = x*x*x * (x * (x * 6 - 15) + 10);

	return lerp(fx, grad1(PERM[i], x), grad1(PERM[ii], x - 1)) * 0.4f;
}

static PyObject *
py_noise1(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x;
	int octaves = 1;
	float persistence = 0.5f;
    float lacunarity = 2.0f;
	int repeat = 1024; // arbitrary
	int base = 0;

	static char *kwlist[] = {"x", "octaves", "persistence", "lacunarity", "repeat", "base", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f|iffii:noise1", kwlist,
		&x, &octaves, &persistence, &lacunarity, &repeat, &base))
		return NULL;
	
	if (octaves == 1) {
		// Single octave, return simple noise
		return (PyObject *) PyFloat_FromDouble((double) noise1(x, repeat, base));
	} else if (octaves > 1) {
		int i;
		float freq = 1.0f;
		float amp = 1.0f;
		float max = 0.0f;
		float total = 0.0f;

		for (i = 0; i < octaves; i++) {
			total += noise1(x * freq, (const int)(repeat * freq), base) * amp;
			max += amp;
			freq *= lacunarity;
			amp *= persistence;
		}
		return (PyObject *) PyFloat_FromDouble((double) (total / max));
	} else {
		PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
		return NULL;
	}
}

static inline float
grad2(const int hash, const float x, const float y)
{
	const int h = hash & 15;
	return x * GRAD3[h][0] + y * GRAD3[h][1];
}

float
noise2(float x, float y, const float repeatx, const float repeaty, const int base)
{
	float fx, fy;
	int A, AA, AB, B, BA, BB;
	int i = (int)floorf(fmodf(x, repeatx));
	int j = (int)floorf(fmodf(y, repeaty));
	int ii = (int)fmodf(i + 1, repeatx);
	int jj = (int)fmodf(j + 1, repeaty);
	i = (i & 255) + base;
	j = (j & 255) + base;
	ii = (ii & 255) + base;
	jj = (jj & 255) + base;

	x -= floorf(x); y -= floorf(y);
	fx = x*x*x * (x * (x * 6 - 15) + 10);
	fy = y*y*y * (y * (y * 6 - 15) + 10);

	A = PERM[i];
	AA = PERM[A + j];
	AB = PERM[A + jj];
	B = PERM[ii];
	BA = PERM[B + j];
	BB = PERM[B + jj];
		
	return lerp(fy, lerp(fx, grad2(PERM[AA], x, y),
							 grad2(PERM[BA], x - 1, y)),
					lerp(fx, grad2(PERM[AB], x, y - 1),
							 grad2(PERM[BB], x - 1, y - 1)));
}

static PyObject *
py_noise2(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x, y;
	int octaves = 1;
	float persistence = 0.5f;
    float lacunarity = 2.0f;
	float repeatx = 1024; // arbitrary
	float repeaty = 1024; // arbitrary
	int base = 0;

	static char *kwlist[] = {"x", "y", "octaves", "persistence", "lacunarity", "repeatx", "repeaty", "base", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ff|iffffi:noise2", kwlist,
		&x, &y, &octaves, &persistence, &lacunarity, &repeatx, &repeaty, &base))
		return NULL;
	
	if (octaves == 1) {
		// Single octave, return simple noise
		return (PyObject *) PyFloat_FromDouble((double) noise2(x, y, repeatx, repeaty, base));
	} else if (octaves > 1) {
		int i;
		float freq = 1.0f;
		float amp = 1.0f;
		float max = 0.0f;
		float total = 0.0f;

		for (i = 0; i < octaves; i++) {
			total += noise2(x * freq, y * freq, repeatx * freq, repeaty * freq, base) * amp;
			max += amp;
			freq *= lacunarity;
			amp *= persistence;
		}
		return (PyObject *) PyFloat_FromDouble((double) (total / max));
	} else {
		PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
		return NULL;
	}
}

static inline float
grad3(const int hash, const float x, const float y, const float z)
{
	const int h = hash & 15;
	return x * GRAD3[h][0] + y * GRAD3[h][1] + z * GRAD3[h][2];
}

float
noise3(float x, float y, float z, const int repeatx, const int repeaty, const int repeatz, 
	const int base)
{
	float fx, fy, fz;
	int A, AA, AB, B, BA, BB;
	int i = (int)floorf(fmodf(x, repeatx));
	int j = (int)floorf(fmodf(y, repeaty));
	int k = (int)floorf(fmodf(z, repeatz));
	int ii = (int)fmodf(i + 1,  repeatx);
	int jj = (int)fmodf(j + 1, repeaty);
	int kk = (int)fmodf(k + 1, repeatz);
	i = (i & 255) + base;
	j = (j & 255) + base;
	k = (k & 255) + base;
	ii = (ii & 255) + base;
	jj = (jj & 255) + base;
	kk = (kk & 255) + base;

	x -= floorf(x); y -= floorf(y); z -= floorf(z);
	fx = x*x*x * (x * (x * 6 - 15) + 10);
	fy = y*y*y * (y * (y * 6 - 15) + 10);
	fz = z*z*z * (z * (z * 6 - 15) + 10);

	A = PERM[i];
	AA = PERM[A + j];
	AB = PERM[A + jj];
	B = PERM[ii];
	BA = PERM[B + j];
	BB = PERM[B + jj];
		
	return lerp(fz, lerp(fy, lerp(fx, grad3(PERM[AA + k], x, y, z),
									  grad3(PERM[BA + k], x - 1, y, z)),
							 lerp(fx, grad3(PERM[AB + k], x, y - 1, z),
									  grad3(PERM[BB + k], x - 1, y - 1, z))),
					lerp(fy, lerp(fx, grad3(PERM[AA + kk], x, y, z - 1),
									  grad3(PERM[BA + kk], x - 1, y, z - 1)),
							 lerp(fx, grad3(PERM[AB + kk], x, y - 1, z - 1),
									  grad3(PERM[BB + kk], x - 1, y - 1, z - 1))));
}

static PyObject *
py_noise3(PyObject *self, PyObject *args, PyObject *kwargs)
{
	float x, y, z;
	int octaves = 1;
	float persistence = 0.5f;
    float lacunarity = 2.0f;
	int repeatx = 1024; // arbitrary
	int repeaty = 1024; // arbitrary
	int repeatz = 1024; // arbitrary
	int base = 0;

	static char *kwlist[] = {"x", "y", "z", "octaves", "persistence", "lacunarity",
		"repeatx", "repeaty", "repeatz", "base", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff|iffiiii:noise3", kwlist,
		&x, &y, &z, &octaves, &persistence, &lacunarity, &repeatx, &repeaty, &repeatz, &base))
		return NULL;
	
	if (octaves == 1) {
		// Single octave, return simple noise
		return (PyObject *) PyFloat_FromDouble((double) noise3(x, y, z, 
			repeatx, repeaty, repeatz, base));
	} else if (octaves > 1) {
		int i;
		float freq = 1.0f;
		float amp = 1.0f;
		float max = 0.0f;
		float total = 0.0f;

		for (i = 0; i < octaves; i++) {
			total += noise3(x * freq, y * freq, z * freq, 
				(const int)(repeatx*freq), (const int)(repeaty*freq), (const int)(repeatz*freq), base) * amp;
			max += amp;
			freq *= lacunarity;
			amp *= persistence;
		}
		return (PyObject *) PyFloat_FromDouble((double) (total / max));
	} else {
		PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
		return NULL;
	}
}

/**
 * Be like:
 * def batch_noise2(
 *     min_x: float, min_y: float,
 *     max_x: float, max_y: float,
 * 	   repeat_x: float, repeat_y: float,
 *     base: float, resolution: float,
 *     callback: Optional[Callable] = None
 * ) -> np.ndarray[np.float32]
 * 
 * Yet, this is a single-thread calculating function. 
 * Once you want to combine the multiple results via multi threads,
 *   how to initialize the seamless mosaicking... I'm still thinking.
 */

static PyObject* py_batch_noise2(PyObject* self, PyObject* args, PyObject* kwargs) {

    // Prepare parameters.
    float min_x, min_y, max_x, max_y;
	float repeat_x = 1024.0f, repeat_y = 1024.0f;
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
        &min_x, &min_y, &max_x, &max_y, &repeat_x, &repeat_y, &base, &resolution,
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

    float step_x = width / resolution;
    float step_y = height / resolution;

	// progress bar for now.
	float total = step_x * step_y;
	float now = 0.0f;

	// Iter.
	for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float x = min_x + i * step_x;
            float y = min_y + j * step_y;
            float val = noise2(x, y, repeat_x, repeat_y, base);
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


static PyMethodDef perlin_functions[] = {
	{"noise1", (PyCFunction) py_noise1, METH_VARARGS | METH_KEYWORDS, 
		"noise1(x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0.0)\n\n"
		"1 dimensional perlin improved noise function (see noise3 for more info)"},
	{"noise2", (PyCFunction) py_noise2, METH_VARARGS | METH_KEYWORDS, 
		"noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0.0)\n\n"
		"2 dimensional perlin improved noise function (see noise3 for more info)"},
	{"noise3", (PyCFunction) py_noise3, METH_VARARGS | METH_KEYWORDS, 
		"noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0, "
			"repeatx=1024, repeaty=1024, repeatz=1024, base=0.0)\n\n"
		"return perlin \"improved\" noise value for specified coordinate\n\n"
		"octaves -- specifies the number of passes for generating fBm noise,\n"
		"defaults to 1 (simple noise).\n\n"
		"persistence -- specifies the amplitude of each successive octave relative\n"
		"to the one below it. Defaults to 0.5 (each higher octave's amplitude\n"
		"is halved). Note the amplitude of the first pass is always 1.0.\n\n"
        "lacunarity -- specifies the frequency of each successive octave relative\n"
        "to the one below it, similar to persistence. Defaults to 2.0.\n\n"
		"repeatx, repeaty, repeatz -- specifies the interval along each axis when \n"
		"the noise values repeat. This can be used as the tile size for creating \n"
		"tileable textures\n\n"
		"base -- specifies a fixed offset for the input coordinates. Useful for\n"
		"generating different noise textures with the same repeat interval"},
	{"batch_noise2", (PyCFunction)py_batch_noise2, METH_VARARGS | METH_KEYWORDS, 
		"batch_noise2(\n"
		"min_x: float, min_y: float, max_x: float, max_y: float, "
		"repeat_x: float = 1024.0, repeat_y: float = 1024.0, base: float = 0.0, \n"
		"resolution: float = 30.0\n"
		")\n\n"
		"Generate a 2D array of Perlin noise values.\n\n"
		"min_x, min_y -- minimum coordinate values.\n"
		"max_x, max_y -- maximum coordinate values.\n"
		"repeat_x, repeat_y, base -- (see noise3 for more info)\n"
		"resolution -- number of samples per unit.\n"
		"for acquiring the progress of noise generating.\n\n"
		"Written via: 月と猫 - LunaNeko.\n"
		"(I just want to generate this in C iteration rather than Python iteration \n"
		"for the iteration in Python is too slow for me.)"
	},
	{NULL}
};

PyDoc_STRVAR(module_doc, "Native-code tileable Perlin \"improved\" noise functions");

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"_perlin",
	module_doc,
	-1,                 /* m_size */
	perlin_functions,   /* m_methods */
	NULL,               /* m_reload (unused) */
	NULL,               /* m_traverse */
	NULL,               /* m_clear */
	NULL                /* m_free */
};

PyObject *
PyInit__perlin(void)
{
	import_array();
    return PyModule_Create(&moduledef);
}

#else

void
init_perlin(void)
{
	Py_InitModule3("_perlin", perlin_functions, module_doc);
}

#endif
