"""
Noise functions for procedural generation of content

Contains native code implementations of Perlin improved noise (with
fBm capabilities) and Perlin simplex noise. Also contains a fast
"fake noise" implementation in GLSL for execution in shaders.

Copyright (c) 2008, Casey Duncan (casey.duncan@gmail.com)

----
Here is the information of `月と猫 - LunaNeko`:

I found that this library is a legacy library and REALLY
need to be remastered for it doesn't have ANY of typing
checking in its functions.

This library is a REMASTERED version of the original library.
Especially for Python calling and type checking.

"""

__version__ = "2.0"

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional

from . import _perlin, _simplex

def snoise2(
        x: float, y: float, 
        octaves: int = 1, 
        persistence: float = 0.5, 
        lacunarity: float = 2.0, 
        repeatx: Optional[float] = None, repeaty: Optional[float] = None, 
        base: float = 0.0
    ) -> float:
    """Generate 2D simplex noise value for specified coordinate.

    Args:
        x: X coordinate.
        y: Y coordinate.
        octaves: Specifies the number of passes, defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.
        repeatx: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeaty: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the noise coordinates. Useful for
            generating different noise textures with the same repeat interval.

    Returns:
        Simplex noise value for the specified 2D coordinate.
    """
    return _simplex.noise2(x, y, octaves, persistence, lacunarity, repeatx, repeaty, base)

def snoise3(
        x: float, y: float, z: float, 
        octaves: int = 1, 
        persistence: float = 0.5,
        lacunarity: float = 2.0
        ) -> float:
    """Generate 3D simplex noise value for specified coordinate.

    Args:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
        octaves: Specifies the number of passes, defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.

    Returns:
        Simplex noise value for the specified 3D coordinate.
    """
    return _simplex.noise3(x, y, z, octaves, persistence, lacunarity)

def snoise4(
        x: float, y: float, z: float, w: float, 
        octaves: int = 1, 
        persistence: float = 0.5, 
        lacunarity: float = 2.0
        ) -> float:
    """Generate 4D simplex noise value for specified coordinate.

    Args:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
        w: W coordinate.
        octaves: Specifies the number of passes, defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.

    Returns:
        Simplex noise value for the specified 4D coordinate.
    """
    return _simplex.noise4(x, y, z, w, octaves, persistence, lacunarity)

def pnoise1(
        x: float, 
        octaves: int = 1, 
        persistence: float = 0.5, 
        lacunarity: float = 2.0, 
        repeat: int = 1024, 
        base: int = 0
        ) -> float:
    """Generate 1D perlin improved noise value for specified coordinate.

    Args:
        x: X coordinate.
        octaves: Specifies the number of passes for generating fBm noise,
            defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.
        repeat: Specifies the interval along axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.

    Returns:
        Perlin "improved" noise value for the specified coordinate.
    """
    return _perlin.noise1(x, octaves, persistence, lacunarity, repeat, base)

def pnoise2(
        x: float, y: float, 
        octaves: int = 1, 
        persistence: float = 0.5,
        lacunarity: float = 2.0, 
        repeatx: float = 1024.0, repeaty: float = 1024.0,
        base: int = 0
        ) -> float:
    """Generate 2D perlin improved noise value for specified coordinate.

    Args:
        x: X coordinate.
        y: Y coordinate.
        octaves: Specifies the number of passes for generating fBm noise,
            defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.
        repeatx: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeaty: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.

    Returns:
        Perlin "improved" noise value for the specified coordinate.
    """
    return _perlin.noise2(x, y, octaves, persistence, lacunarity, repeatx, repeaty, base)

def pnoise3(
        x: float, y: float, z: float, 
        octaves: int = 1, persistence: float = 0.5,
        lacunarity: float = 2.0, 
        repeatx: int = 1024, repeaty: int = 1024, repeatz: int = 1024, 
        base: int = 0
        ) -> float:
    """Generate 3D perlin improved noise value for specified coordinate.

    Args:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
        octaves: Specifies the number of passes for generating fBm noise,
            defaults to 1 (simple noise).
        persistence: Specifies the amplitude of each successive octave relative
            to the one below it. Defaults to 0.5 (each higher octave's amplitude
            is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: Specifies the frequency of each successive octave relative
            to the one below it, similar to persistence. Defaults to 2.0.
        repeatx: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeaty: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeatz: Specifies the interval along z axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.

    Returns:
        Perlin "improved" noise value for the specified coordinate.
    """
    return _perlin.noise3(x, y, z, octaves, persistence, lacunarity, repeatx, repeaty, repeatz, base)

# Added my homebrew batch calculating function.
def batch_pnoise2(
        min_x: float, min_y: float,
        max_x: float, max_y: float,
        repeat_x: float = 1024.0, repeat_y: float = 1024.0,
        base: float = 0.0,
        resolution: float = 30.0,
        callback: Optional[Callable] = None
    ) -> NDArray[np.float32]: 
    """Generate a 2D array of Perlin noise values.

    Args:
        min_x: Minimum X coordinate value.
        min_y: Minimum Y coordinate value.
        max_x: Maximum X coordinate value.
        max_y: Maximum Y coordinate value.
        repeat_x: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeat_y: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.
        resolution: Number of samples per unit.
        callback: Optional callback function for acquiring the progress of noise generating.

    Returns:
        2D numpy array of Perlin noise values.
    """
    return _perlin.batch_noise2(
        min_x, min_y,
        max_x, max_y,
        repeat_x, repeat_y,
        base,
        resolution,
        callback
    )

def batch_pnoise3(
        min_x: float, min_y: float, min_z: float,
        max_x: float, max_y: float, max_z: float,
        repeat_x: float = 1024.0, repeat_y: float = 1024.0, repeat_z: float = 1024.0,
        base: float = 0.0,
        resolution: float = 30.0,
        callback: Optional[Callable] = None
    ) -> NDArray[np.float32]: 
    """Generate a 2D array of Perlin noise values.

    Args:
        min_x: Minimum X coordinate value.
        min_y: Minimum Y coordinate value.
        max_x: Maximum X coordinate value.
        max_y: Maximum Y coordinate value.
        repeat_x: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeat_y: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.
        resolution: Number of samples per unit.
        callback: Optional callback function for acquiring the progress of noise generating.

    Returns:
        3D numpy array of Perlin noise values.
    """
    return _perlin.batch_noise3(
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        repeat_x, repeat_y, repeat_z,
        base,
        resolution,
        callback
    )

def batch_snoise2(
        min_x: float, min_y: float,
        max_x: float, max_y: float,
        repeat_x: Optional[float] = None, repeat_y: Optional[float] = None,
        base: float = 0.0,
        resolution: float = 30.0,
        callback: Optional[Callable] = None
    ) -> NDArray[np.float32]: 
    """Generate a 2D array of Simplex noise values.

    Args:
        min_x: Minimum X coordinate value.
        min_y: Minimum Y coordinate value.
        max_x: Maximum X coordinate value.
        max_y: Maximum Y coordinate value.
        repeat_x: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeat_y: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.
        resolution: Number of samples per unit.
        callback: Optional callback function for acquiring the progress of noise generating.

    Returns:
        2D numpy array of Simplex noise values.
    """
    return _simplex.batch_noise2(
        min_x, min_y,
        max_x, max_y,
        repeat_x, repeat_y,
        base,
        resolution,
        callback
    )

def batch_snoise3(
        min_x: float, min_y: float, min_z: float,
        max_x: float, max_y: float, max_z: float,
        repeat_x: float = 1024.0, repeat_y: float = 1024.0, repeat_z: float = 1024.0,
        base: float = 0.0,
        resolution: float = 30.0,
        callback: Optional[Callable] = None
    ) -> NDArray[np.float32]: 
    """Generate a 3D array of Simplex noise values.

    Args:
        min_x: Minimum X coordinate value.
        min_y: Minimum Y coordinate value.
        min_z: Minimum Z coordinate value.
        max_x: Maximum X coordinate value.
        max_y: Maximum Y coordinate value.
        max_z: Maximum Z coordinate value.
        repeat_x: Specifies the interval along x axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeat_y: Specifies the interval along y axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        repeat_z: Specifies the interval along z axis when the noise values repeat.
            This can be used as the tile size for creating tileable textures.
        base: Specifies a fixed offset for the input coordinates. Useful for
            generating different noise textures with the same repeat interval.
        resolution: Number of samples per unit.
        callback: Optional callback function for acquiring the progress of noise generating.

    Returns:
        3D numpy array of Simplex noise values.
    """
    return _simplex.batch_noise3(
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        repeat_x, repeat_y, repeat_z,
        base,
        resolution,
        callback
    )
