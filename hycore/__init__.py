"""
A functional data structure for creating, managing and analysing hyperspectral drillcore datasets using python and hylite.
"""

from .coreshed import *

import os
from pathlib import Path
import shutil

sandbox = None

def loadShed( path ):
    """
    Load a shed Object
    :param path: Open the specified Shed.
    :return: A hycore.Shed instance.
    """

    # check shed exists and is a directory
    pth = os.path.splitext(path)[0] + ".shed"
    if not os.path.exists(pth) and os.path.isdir(pth):
        raise FileNotFoundError("%s is not a valid hycore.Shed" % path)
    name = os.path.basename( os.path.splitext( path )[0] )

    # load header (if one exists)
    from hylite import io
    header = None
    hdr = os.path.splitext(path)[0] + ".hdr"
    if os.path.exists(hdr):
        header = io.loadHeader( hdr )

    # create and return Shed instance
    from hycore.coreshed import Shed
    S = Shed(name, os.path.dirname( pth ) , header )

    # load holes and boxes to ensure correct dtype (Hole or Box, not HyCollection)
    _ = S.getBoxes()

    return Shed(name, os.path.dirname( pth ) , header )

def get_sandbox( fill=False, vis=False, mosaic=False ):
    """
    Return a sandbox directory on the local file system for reading / writing temporary files.
    :param fill: True if a dummy hyperspectral core database should be added to the sandbox directory. If this is
                  set, a shed will be returned rather than the directory path.
    :param vis: True if .png previews should be generated in the shed.
    :param mosaic: True if .png mosiacs should be generated for each drillcore.
    :return: A Path defining the sandbox directory or a Shed instance (if fill = True).
    """
    global sandbox
    if sandbox is None:
        sandbox = Path( os.path.join(os.getcwd(), 'sandbox') )
    os.makedirs( sandbox, exist_ok = True ) # if we are getting it, the sandbox dir should exist!

    from hycore.generator import genTray
    from hycore.coreshed import Shed

    if fill:
        # create a dummy coreshed!
        S = Shed('eldorado', sandbox)

        # add fake holes and data
        h1 = [False, False, True]  # True / False are lithology codes
        h2 = [True, False, True]
        h3 = [False]  # include a hole with only one tray for testing purposes
        for h, n, c, b in zip([h1, h2, h3], [4, 5, 4], ['H01', 'H03', 'H02'], [1.0, 0.4, 1.0]):
            trays = [genTray(n, l) for l in h]
            ids = [i + 1 for i in range(len(h))]
            for (timg, A, mask), i in zip(trays, ids):
                # add Fenix image
                timg.data[..., timg.get_band_index(2180.):timg.get_band_index(
                    2210.)] *= b  # adjust brightness of 2200 nm band to provide visual difference between different holes
                t = S.addHSI('FENIX', c, '%03d' % i, timg)

                # add mask
                t.mask = hylite.HyImage(mask.astype(np.uint8))

                # add a result
                from hylite.analyse import band_ratio
                R = band_ratio(t.FENIX, 2230., 2200.)
                G = band_ratio(t.FENIX, 2190., 2200.)
                B = band_ratio(t.FENIX, 2140., 2160.)
                stack = hylite.HyImage(np.clip(np.dstack([(R.data - 1.0) / 0.2,
                                                          (G.data - 1.) / 0.2,
                                                          (B.data - 1.0) / 0.2]), 0, 1))
                stack.data = (stack.data * 255).astype(np.uint8)  # convert to RGB
                t.results.BR_Clays = stack

                def saveLegend(red, green, blue, path):
                    import matplotlib.pyplot as plt
                    tr = [np.nan, np.nan, np.nan]
                    plt.figure(figsize=(4, 0.5))
                    plt.fill(tr, tr, label=red, color='r')
                    plt.fill(tr, tr, label=green, color='g')
                    plt.fill(tr, tr, label=blue, color='b')
                    plt.legend(ncol=3, loc='center')
                    plt.axis('off')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(path, dpi=300)
                    plt.close()

                saveLegend(r"R= ($\frac{2230}{2200}$)",
                           r"G= ($\frac{2190}{2200}$)",
                           r"B= ($\frac{2140}{2160}$)",
                           os.path.join(t.results.getDirectory(), 'LEG_Clays.png'))

                # add fake image as though from another sensor
                lwir = timg.copy()
                lwir.data = 1. - lwir.data
                lwir.set_wavelengths(timg.get_wavelengths() * 10 - 13000)
                t = S.addHSI('LWIR', c, '%03d' % i, lwir)
                t.start = (i - 1) * n
                t.end = i * n
                t.save()
                t.free()

        # add a box that is non-contiguous (has a gap)
        t = S.addHSI('FENIX', 'H01', '004', timg)
        t = S.addHSI('LWIR', 'H01', '004', lwir)
        t.start = 15
        t.end = 20
        t.mask = hylite.HyImage(mask.astype(np.uint8))
        t.results.BR_Clays = stack
        t.save()
        t.free()
        saveLegend(r"R= ($\frac{2230}{2200}$)",
                   r"G= ($\frac{2190}{2200}$)",
                   r"B= ($\frac{2140}{2160}$)",
                   os.path.join(t.results.getDirectory(), 'LEG_Clays.png'))

        if vis:
            for b in S.getBoxes():
                b.updateVis(qaqc=False, thumb='FENIX')

        if mosaic:
            from hycore.templates import unwrap_tray, Template, Canvas, compositeStack
            for i, h in enumerate(S.getHoles()):
                # get pole mosaic of core
                T = h.getTemplate(method='pole', res=2e-3)

                # apply template to generate strips for various products
                for f in ['FENIX.png', 'LWIR.png', 'BR_Clays.png']:
                    m = T.apply(f, (0,1,2))
                    h.results.set(f, m)
                h.results.save()
                h.save()
                h.free()

            # add some example annotations to the final hole
            h.annotate("Vein 1", "I think there's gold here!", depth_from=0.5, group='Gold')
            h.annotate("Vein 2", "No gold here.", depth_from=1, depth_to=2, group='Gold')
            h.annotate("Fault 1", "What a bloody mess!", depth_from=0.5, depth_to=2, group='Faults')
            h.annotate("Fault 2", "https://en.wikipedia.org/wiki/San_Andreas_Fault",
                       depth_from=2.1, depth_to=2.1, group='Faults', type='link')
            h.save()

        S.save()
        S.free()
        return S
    else:
        return sandbox


def set_sandbox( path ):
    """
    Set the sandbox directory for hycore to use.
    :param path: A path to the new sandbox directory
    :return: None
    """
    global sandbox
    sandbox = sandbox

def empty_sandbox():
    """
    Remove the sandbox directory (if it exists).
    :return: None
    """
    shutil.rmtree( get_sandbox() )

