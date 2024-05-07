"""
This defines wrapper classes that encapsulate underlying HyCollection objects to represent different types
of data stored associated with drillcore scanning projects. Generally these are stored in a top-level "Shed" object.
"""
import os
import hylite
from hylite import HyCollection
from hylite import io
import numpy as np
from natsort import natsorted
import uuid
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import shutil

def __getAs__( root, name, type, *, create=False ):
    """
    Find a HyCollection child and return it as the specified hycore type.
    :param root: HyCollection object to search.
    :param name: String name of the attribute to find
    :param type: type to cast HyCollection too if needed. Can be Shed, Hole, Tray, or Sample.
    :param create: if True, the relevant HyCollection will be created if it does not exist.
    :return: The subcollection in the desired type.
    """

    try:
        # try and get the object
        H = root.get(name)

        # update its type if needed
        if isinstance(H, hylite.HyCollection):
            H2 = type(H.name, H.root, header=H.header)
            for a in H.getAttributes(ram_only=True, file_formats=False):
                H2.set(a, H.get(a)) # copy attributes in RAM across
        root.set(name, H2) # update object in root to match
    except:
        if create:
            print("Creating %s" % name)
            root.set(name, type(name, root.getDirectory()))
        else:
            assert False, "Error - an object called %s does note exist in %s" % (name, root.name)

    H = root.get(name)
    assert isinstance(H, type), "Error - attribute %s exists but it is the wrong type (%s not %s)" % (
    name, type(H), type)
    return H

def __getSub__(C, ctype=None):
    """
    Return subcollections of a HyCollection.
    :param C (HyCollection): The collection to search.
    :param ctype (str): Only return subcollections of the specified types.
    :return: A list of HyCollection instances. Note that this DOES NOT cast them to the relevant hycore type.
    """
    out = []
    for a in C.query(ext_pattern=['.hyc', 'HyCollection', 'Hole', 'Box', 'Sample']):
        a = C.get(a)
        if ctype is not None:
            #attr = a.getAttributes(ram_only=True)
            #if 'ctype' in attr:
            #if hasattr(a, 'ctype'):
            try:
                if a.ctype == ctype:
                    out.append(a)
            except:
                continue # ctype not defined, so don't match
    return out

ternary = {'RGB' : (0, 1, 2), 'FENIX' : hylite.SWIR, 'FX50' : hylite.MWIR, 'LWIR' : hylite.LWIR }
"""
A global dict containing standard ternary mappings used for visualising each sensors while visualising
box data. This dict can be modified or added to as needed, though keys should all be strings and values
should all be tuples of length 3 containing (red_band, green_band, blue_band), following hylite band notation 
such that float values are interpreted as wavelengths and int values are interpreted as band indices.
"""

ternary_clip = (2,98)
""" A tuple of length two containing the vmin and vmax percentiles used to define black and white when generating
    visualisations. Default is (2, 98)."""

class BaseCollection( HyCollection ):
    """
    HyCollection but with "vis" and "results" internal collections.
    """
    def __init__(self, name, root, *, ctype, ext='.hyc', header=None):
        super().__init__(str(name), str(root), header)
        self.ctype = ctype
        self.ext = ext
        # self.vis = self.addSub('vis')  # HyCollection containing preview imagery for visualisation
        self.results = self.addSub('results')  # HyCollection containing results data

        # make sure output directory and header with basic metadata exists
        #os.makedirs(self.getDirectory(), exist_ok=True)
        #if not os.path.exists( os.path.splitext(self.getDirectory())[0] + ".hdr"):
        #    io.saveHeader( os.path.splitext(self.getDirectory())[0] + ".hdr", self.header )
        #    # self.save()

    def get(self, name):
        """
        Get a dataset or result in this box.

        :param name: The variable to get. This can optionally include a file extension (to force e.g., PNG images).
        :return: The retrieved data.
        """

        if '.' in name:
            n,e = os.path.splitext(name)
            if os.path.exists( os.path.join(self.getDirectory(), name ) ):
                self.set(n, io.load( os.path.join(self.getDirectory(), name ) )) # load and return
                return self.get( n ) # this may be a Box, Hole etc.
            elif os.path.exists( os.path.join(self.results.getDirectory(), name )):
                self.results.set(n, io.load(os.path.join(self.results.getDirectory(), name )))  # load and return
                return self.results.get(n) # this will always be an object in the results HyCollection.
            else:
                raise AttributeError(name)
        else:
            try:
                return super().get(name) # look in this collection
            except AttributeError:
                return self.results.get(name) # look in results collection!

class Shed( BaseCollection ):
    """
    A collection of datasets (drillholes) and processing workflows associated with a project. A shed also hosts the
    worker threads used for parallel processing. Think of everything performed in your average core shed (except core
    cutting - that is nasty work!).
    """

    """ Band combinations to use for ternary visualisations"""
    def __init__(self, name, root, header=None):
        super().__init__(name, root, ext='.shed', ctype = "hycore.coreshed.Shed", header=header)
        self.getBoxes() # run this immediately to ensure types convert correctly

        # self.holes = self.addSub('holes') # HyCollection containing hole data
        # self.samples = self.addSub('samples') # HyCollection for containing (arbitrary) sample data

    def addHSI(self, sensor, hole, boxID, image ):
        """
        Adds a hyperspectral image to this Shed.
        :param sensor: The name of the sensor that captured this HSI image (e.g. 'fenix').
        :param hole: The name of the hole that this box belongs too.
        :param boxID: string defining the unique id of the box.
        :param image: A HyImage instance containing the hyperspectral data.
        :return: A Box object encapsulating this data.
        """
        H = self.getHole(hole, create=True )
        return H.addHSI( sensor, boxID, image )

    def getHole(self, name, *, create=False):
        """
        Get the corresponding hole from this coreshed.
        :param name: string containing the name of the hole.
        :param create: True if a hole should be created if one with the specified name does not exist. Default is False.
        :return: A Hole object.
        """
        return __getAs__( self, name, Hole, create=create)

    def getBox(self, hole, box, *, create=False):
        """
        Get the corresponding box from this coreshed.
        :param hole: The hole (name) this box is in.
        :param box: The box name.
        :param create: If True, the box (and hole) will be created if it doesn't exist
        :return: A Box object.
        """
        hole = self.getHole( hole, create=create )
        return hole.getBox( box, create=create)

    def getHoles(self):
        """
        :return: A list of Hole objects stored in this Shed.
        """
        return [__getAs__( self, h.name, Hole ) for h in __getSub__( self, ctype="hycore.coreshed.Hole" ) ]

    def getBoxes(self):
        """
        :return: A list of all the boxes in this shed (from all holes)
        """
        boxes = []
        for h in self.getHoles():
            boxes += h.getBoxes()
        return boxes

    def getTemplate(self, method='fence', *, from_depth=None, to_depth=None, res: float = 1e-3,
                          holes: list = None, depth_offsets : dict = {}, **kwds ):
        """
        Get a template that mosaics boreholes in this shed. Two different layouts are possible:

        pole layout: lay individual core sticks end to end, as though in the arrangement they were drilled.

        core 1: [  ==== ===    ======= ========  ===== ==== ]   Note gaps are added between core blocks to ensure scale is real.
        core 2: [  ======   == ================  ========== ]
        core 3: [  ==== =====   === ===========  ========== ]
                   --------------------------------> downhole direction

        fence layout: lay individual core sticks one on-top the other, as though arranged in stacked core boxes

            core 1       core 2         core 3
    1  - | ======== | | =========| | ========== |
         | ======== | | =========| | ========== |
    2  - | ======== | | ======   | | =====      |
         | ======== |     gap      | ========== |
    3  - | ======   | | =========| | ========== |
         | ======== | | =========| | ========== |

        :param method: The mosaic method. Options are 'pole' and 'fence' (default).
        :param from_depth: The top depth of the template view area, or None to include all depths.
        :param to_depth: The lower depth of the template view area, or None to include all depths.
        :param res: The pixel size in meters. Default is 1 mm (1e-3 m). See hycore.templates.Canvas for further details.
                    N.B. this resolution is only used in 'pole' layout. Fence layout will add a gap between non-contiguous boxes.
        :param holes: Names of the holes to plot (in order!). If None (default) then all holes are plotted.
        :param depth_offsets: A dictionary containing depth values to be subtracted from sub-templates with matching
                                group names. Useful for e.g., plotting boreholes relative to a marker horizon rather than
                                in absolute terms.
        :param kwds: All keywords are passed to Box.getTemplate()
        :return: A Template for this hole.
        """
        from hycore.templates import Template, Canvas, unwrap_tray  # import here as otherwise we get a circular import error

        # get holes if needed
        if holes is None:
            holes = [h.name for h in self.getHoles()]

        # build canvas
        C = Canvas()
        for h in holes:
            for b in self.getHole(h).getBoxes():
                if 'fence' in method.lower():
                    T = b.getTemplate(axis=1, **kwds)
                elif 'pole' in method.lower():
                    T = b.getTemplate(axis=0, **kwds) # unwrap box and get template
                if T is not None:
                    C.add(h, T) # add template to canvas (N.B. this will only have one group [ this core ] )

        # build and return template
        if 'fence' in method.lower():
            return C.vfence(from_depth=from_depth, to_depth=to_depth,
                            groups=holes, depth_offsets=depth_offsets)
        elif 'pole' in method.lower():
            return C.hpole(from_depth=from_depth, to_depth=to_depth, res=res, scaled=True,
                           groups=holes, depth_offsets=depth_offsets)
        else:
            assert False, "Error, %s is an unknown method. Try 'pole' or 'fence'."%method

    def getSensors(self):
        """
        Return a list of (all) sensors in this hole.
        :return: A list of sensor present in one or more of the boxes.
        """
        sensors = []
        for b in self.getBoxes():
            sensors += list(b.getSensors())
        return natsorted( list(set(sensors)) )

    def updateMosaics(self, **kwargs):
        """
        Calls updateMosaics( ... ) on each hole in this shed. See Hole.updateMosaics(...) for details and description of arguments.
        :param kwargs: All keywords and arguments are passed to hole.updateMosaics( ... ).
        """
        for h in self.getHoles():
            h.updateMosaics( **kwargs )

    def createAboutMD(self, author_name, date=None):
        """
        Create a dummy `about.md` file in the shed directory. Useful for subsequently adding metadata
        that describes this dataset.
        :param author_name: A string containing the name of the author(s) of this dataset.
        :param date: A string containing the date to assign to this dataset. If None (default) then todays date is used.
        :return: A path to the created markdown file.
        """

        # get relevant metadata
        nholes = len( self.getHoles() )
        nboxes = len( self.getBoxes() )
        nmeters = int(np.sum( [h.scannedLength() for h in self.getHoles()]))
        span = int(np.sum( [h.scannedSpan() for h in self.getHoles() ] ))
        coverage = int(100*(nmeters/span))
        if date is None:
            from datetime import date
            date = date.today()

        # build markdown file
        txt = "# {shedname}\n\n".format(shedname=self.name.capitalize())
        txt += "---\n\n"
        txt += "This shed has {nholes} drill holes, totalling {nboxes} boxes that " \
               "cumulatively contain {nmeters} meters of scanned cores.\n ".format(nholes=nholes, nboxes=nboxes, nmeters=nmeters)
        txt += "These cover a total span of {span} meters, representing a coverage of {coverage} percent. \n\n".format(span=span, coverage=coverage)
        txt += "{shedname} is a really cool place and you should extend this description to provide lots of relevant info.\n\n".format(shedname=self.name.capitalize())
        txt += "---\n\n"
        txt += "**Author:** *{author}*\n\n".format(author=author_name)
        txt += "**Date:** *{date}*\n\n".format(date=date)

        # save it
        with open( os.path.join( self.getDirectory(), 'about.md' ), 'w') as f:
            f.write(txt)

    def exportQuanta(self, *, path : str = None, holes : list = None, 
                     sensors : list = None, previews : bool = True,  results : list = [], 
                     force_recalc : bool = False, crop : bool =False, 
                     clean : bool =False, ss : int = 1, **kwargs):
        """
        Export quantized versions of the hyperspectral data in each box in the specified holes
        in this shed to a PNG-based database. This represents a minimal lossy compressed version of the
        hyperspectal data in this shed, that can be useful for generating overview statistics and running
        some types of hyperspectral analyses.

        :param name: The path of the output directory. If None (default) then a directory will be created adjacent to the shed directory.
        :param holes: A list of the holes to export. If None (default) then all holes are used.
        :param sensors: A list of sensors to export. If None (default), then `self.getSensors(..)` is used to find all sensors.
        :param previews: True if preview PNGs for each sensor should be exported too (if they exist).
        :param results: A list of results names (PNGs in box results folders) to export. Defaults to an empty list (no results).
        :param force_recalc: If True, quantized representations of each box will be recomputed. If not, they will only
                             be computed if the do not already exist.
        :param crop: If True, crop images to extent of masked area to further reduce size. Default is False.
        :param clean: Remove all non-png files from outputs (header files etc.). The results can be good for visualisation
                      (e.g., it is used by hywizz), but cannot be used to reconstruct the original data.
        :param ss: Subsampling step if exported images should be downsampled (e.g. to save space). Default is 1 (no downsampling).
        :param kwargs: All other arguments are passed to `Box.quantize`.
        :return: A new Shed instance containing the compressed data.
        """

        # get holes
        if holes is None:
            holes = self.getHoles()

        # get sensors
        if sensors is None:
            sensors = self.getSensors()

        # create output shed
        if path is None:
            S = Shed(self.name + "_qz", os.path.dirname(self.getDirectory() ) )
        else:
            S = Shed(os.path.basename(path), os.path.dirname(path) )
        if os.path.exists( os.path.splitext( S.getDirectory() )[0] ): # delete "clean" directory if needed
            shutil.rmtree( os.path.splitext( S.getDirectory() )[0] )

        # do compression
        for h in holes:
            for b in tqdm(h.getBoxes(), desc="Compressing %s"%h.name, leave=False ):

                # get mask for cropping (if needed)
                if crop and hasattr(b, "mask"):
                    from hycore.templates import get_bounds
                    xmin, xmax, ymin, ymax = get_bounds(b.mask)
                
                # create output box to store data in
                _h = Hole(h.name, S.getDirectory())
                _b = Box(b.name, _h.getDirectory(), header=b.header)
                _s = _b.addSub("spectra")

                for s in sensors:
                    try: # sometimes fails when e.g. some sensors do not exist in some boxes etc.
                        # get band count
                        hdr = io.loadHeader( os.path.join( b.getDirectory(), s + ".hdr" ) )
                        bands = len(hdr.get_wavelengths())
                        if bands > 4: # no point computing quanta for standard images
                            
                            if force_recalc: # force recalculation of quanta
                                lib, idx = b.quantize(sensors=[s], **kwargs)

                            # Does quantised library already exist? If not, create it
                            if not (os.path.exists( os.path.join(b.getDirectory(), 'spectra.hyc/%s_idx.hdr'%s) ) and \
                                    os.path.exists(os.path.join(b.getDirectory(), 'spectra.hyc/%s_lib.hdr' % s))):
                                        lib,idx = b.quantize(sensors=[s], **kwargs)
                            
                            # load quantised index and library
                            lib = hylite.io.load(os.path.join(b.getDirectory(), 'spectra.hyc/%s_lib.hdr' % s))
                            idx = hylite.io.load(os.path.join(b.getDirectory(), 'spectra.hyc/%s_idx.hdr' % s))

                            # normalise spectra in lib to reduce blocky compressed spectra artefacts
                            lib.data = lib.data.astype(np.float32) / np.nanmax(lib.data, axis=((1,2)))[:,None,None] # force max value in each library spectra to 1
                            lib.data = np.nan_to_num( lib.data, nan = 0, posinf=0, neginf=0 ) # nans are bad!
                            idx.data = np.nan_to_num( idx.data, nan = 0, posinf=0, neginf=0 ) # nans are bad!
                            
                            # convert to uint8 so they'll be stored as PNG files
                            if (np.min(idx.data) >= 0) and (np.max(idx.data) <= 255 ):
                                idx.data = (np.clip( idx.data, 0, 255) * 255 ).astype(np.uint8)
                                mx = max( np.nanmax( lib.data ), 1.0 ) # normalising factor
                                lib.data = np.clip( np.transpose( lib.data, (0,2,1) ) / mx * 255, 0, 255 ).astype(np.uint8)
                            else:
                                print("Error - box %s has too many (%d) quanta. Skipping." % (b.getFullName(), np.max(idx.data)))

                            # crop
                            if crop and hasattr(b, "mask"):
                                idx.data = idx.data[ xmin:xmax, ymin:ymax, : ]
                            if ss > 1: # subsample
                                idx.data = idx.data[::ss, ::ss, :]
                            
                            # store data
                            _s.set(s+"_idx", idx )
                            _s.set(s+"_lib", lib )
                    except Exception as e:
                        print("Warning: could not quantise %s in %s: %s"%(s, b.getFullName(), str(e)))

                    # copy preview data?
                    if previews:
                        try:
                            pth = os.path.join( b.getDirectory(), s+".png" )
                            if os.path.exists( pth ):
                                prv = io.load(pth)
                                if crop and hasattr(b, "mask"): # crop if needed
                                    prv.data = prv.data[ xmin:xmax, ymin:ymax, : ]
                                if ss > 1: # subsample
                                    prv.data = prv.data[::ss, ::ss, :]
                                _b.set(s, prv)
                        except Exception as e:
                            print("Warning: could not copy preview %s in %s: %s"%(s, b.getFullName(), str(e)))
                    b.free()

                # copy results data?
                for r in results:
                    p = os.path.join(b.results.getDirectory(), os.path.splitext( r )[0])+".png"
                    if os.path.exists(p):
                        try:
                            im = io.load(p)
                            if crop and (im.xdim() == b.mask.xdim()) and (im.ydim() == b.mask.ydim()):
                                im.data = im.data[ xmin:xmax, ymin:ymax, : ]
                            if ss > 1: # subsample
                                im.data = im.data[::ss, ::ss, :]
                            _b.results.set( os.path.splitext(r)[0], im )
                        except Exception as e:
                            print("Warning: could not copy result %s in %s: %s"%(r, b.getFullName(), str(e)))
                    
                # copy thumb?
                p = os.path.join(b.getDirectory(), "thumb.png")
                if os.path.exists(p):
                    _b.set( 'thumb', io.load(p) )

                # save and free memory
                if not clean:
                    _b.sensor_list = np.array(sensors) # preserve sensors list
                _h.save()
                _b.save()
                _b.free()
                _h.free()

        # clean?
        if clean:
            for hdr in glob.glob( os.path.join( S.getDirectory() + "/**/*.hdr"), recursive=True):
                os.remove( hdr )
            c = glob.glob( os.path.join( S.getDirectory() + "/**/*.hyc"), recursive=True)
            while len(c) > 0:
                os.rename(c[0], os.path.splitext(c[0])[0] )
                c = glob.glob( os.path.join( S.getDirectory() + "/**/*.hyc"), recursive=True)
            os.rename( S.getDirectory(), os.path.splitext( S.getDirectory() )[0] )

class Hole( BaseCollection ):
    """
    A collection of boxes and samples taken from one drillhole, along with localisation information (drillstring).
    """
    def __init__(self, name, root, header=None):
        super().__init__(name, root, ctype = "hycore.coreshed.Hole", header=header)

    def addHSI(self, sensor, boxID, image):
        """
        Adds a hyperspectral image to a box in this borehole.
        :param sensor: The name of the sensor that captured this HSI image (e.g. 'fenix').
        :param boxID: string defining the unique id of the box.
        :param image: A HyImage instance containing the hyperspectral data.
        :return: A Box object encapsulating this data.
        """
        box = self.getBox(boxID, create=True)
        box.addHSI( sensor, image )
        return box

    def getBox(self, boxID, *, create=False):
        """
        Get the corresponding box from this box.
        :param boxID: string defining the unique id of the box.
        :param create: True if a box should be created if one with the specified name does not exist. Default is False.
        :return: A Box object.
        """
        if boxID[0] == 'b':
            return __getAs__(self, '%s'%boxID, Box, create=create)
        else:
            return __getAs__(self, 'b%s'%boxID, Box, create=create)

    def getBoxes(self):
        """
        :return: A list of Box objects stored in this Hole, sorted from top (low depth) to bottom (high depth)
        """

        # get boxes
        boxes = [__getAs__( self, h.name, Box ) for h in __getSub__( self, ctype="hycore.coreshed.Box" ) ]

        # get corresponding depths
        depths = []
        for b in boxes:
            if b.start is not None:
                depths.append(b.start)
            else:
                depths.append(0)

        # return sorted boxes
        idx = np.argsort(depths)
        return [boxes[i] for i in idx]

    def getTemplate(self, method='fence', *, from_depth=None, to_depth=None, res: float = 1e-3, **kwds ):
        """
        Get a template that mosaics this borehole. Two different layouts are possible:

        pole layout: lay individual core sticks end to end, as though in the arrangement they were drilled.

        [  ==== ===    ======= ========  ===== ==== ]   Note gaps are added between core blocks to ensure scale is real.

        fence layout: lay individual core sticks one on-top the other, as though arranged in stacked core boxes

            core 1
    1  - | ======== |
         | ======== |
    2  - | ======== |
         | ======== |
    3  - | ======   |
         | ======== |

        :param method: The mosaic method. Options are 'pole' and 'fence' (default).
        :param from_depth: The top depth of the template view area, or None to include all depths.
        :param to_depth: The lower depth of the template view area, or None to include all depths.
        :param res: The pixel size in meters. Default is 1 mm (1e-3 m). See hycore.templates.Canvas for further details.
                    N.B. this resolution is only used in 'pole' layout. Fence layout will add a gap between non-contiguous boxes.
        :param kwds: All keywords are passed to Box.getTemplate()
        :return: A Template for this hole.
        """
        from hycore.templates import Template, Canvas, unwrap_tray  # import here as otherwise we get a circular import error

        C = Canvas()
        if 'fence' in method.lower():
            for b in self.getBoxes():
                T = b.getTemplate(axis=1, **kwds)
                if T is not None:
                    C.add(self.name, T) # add template to canvas (N.B. this will only have one group [ this core ] )
            return C.vfence(from_depth=from_depth, to_depth=to_depth)
        elif 'pole' in method.lower():
            for b in self.getBoxes():
                T = b.getTemplate(axis=0, **kwds) # unwrap box and get template
                if T is not None:
                    C.add(self.name, T) # add template to canvas (N.B. this will only have one group [ this core ] )
            return C.hpole(from_depth=from_depth, to_depth=to_depth, res=res)
        else:
            assert False, "Error, %s is an unknown method. Try 'pole' or 'fence'."%method

    def scannedLength(self):
        """
        Return cumulative length of all boxes scanned.
        """
        return sum([b.length() for b in self.getBoxes()])

    def scannedSpan(self):
        """
        Return distance spanned between deepest and shallowest box. Note that this length will include gaps (unlike
        scannedLength).
        """
        boxes = self.getBoxes()
        start = min( [b.start for b in boxes if b.start is not None] )
        end = max([b.end for b in boxes if b.end is not None])
        return end - start

    def getSensors(self, boxes=True, samples=False):
        """
        Return a list of (all) sensors in this hole.

        :param boxes: Include sensors covering core boxes. Default is True
        :param samples: Include sensors covering samples. Default is False.
        :return: A list of sensor present in one or more of the boxes / samples.
        """
        sensors = []
        if boxes:
            for b in self.getBoxes():
                sensors += list(b.getSensors())
        if samples:
            for s in self.getSamples():
                sensors += list(s.getSensors())
        return natsorted( list(set(sensors)) )

    def updateMosaics(self, pole=True, fence=True, files=["*.png"], vb=True, outline=0.2, apply_kwargs={}, **kwds):
        """
        Construct mosaics for all data products / visualisations in this drillcore.
        :param pole: If True, pole mosaics will be created and stored in hole.results.pole.
        :param fence: If True, pole mosaics will be created and stored in hole.results.fence.
        :param files: A list of glob queries used to find files to mosaic.  Default is *.png.
        :param vb: True if a progress bar should be created, as the mosaic creation can take quite some time!
        :param outline: An outline colour to highlight masked vs unmasked areas, or None to disable. Default is 0.2.
        :param apply_kwargs: Arguments to pass to Template.apply( ... ). E.g., xstep and ystep if smaller mosaics are needed.
        :keywords: Keyword arguments are passed to hole.getTemplate(...) to specify e.g., depth range or resolution.
        """

        # find files to mosaic
        names = set()
        for q in files:
            for b in self.getBoxes():
                names.update(
                    set([p.name for p in Path(b.getDirectory()).rglob(q)]))  # add any new files that match query

        # get settings for templating
        bands = apply_kwargs.pop('bands', (0,1,2))
        strict = apply_kwargs.pop('strict', False)

        # build templates
        loop = zip( [pole, fence], ['pole', 'fence'] )
        T = []
        O =[]
        if vb:
            loop = tqdm(loop, desc="Building templates for %s" % self.name, leave=False, total=2)
        for b,n in loop:
            if b:
                O.append(self.results.addSub(n))
                T.append(self.getTemplate(n, **kwds))
                O[-1].template = T[-1].toImage()
                O[-1].save()
                O[-1].free()

        # fill mosaics
        loop = names
        if vb:
            loop = tqdm(names, desc="Filling mosaics for %s" % self.name, leave=False)
        for n in loop:
            for _T, _O in zip(T, O):
                apply_kwargs['outline'] = outline
                img = _T.apply(n, bands, strict, **apply_kwargs)
                _O.set(n, img)
                _O.save()
                _O.free()

    def annotate(self, name : str, value : str, depth_from : float, depth_to : float = None, type : str = 'note', group : str = 'notes'):
        """
        Add an annotation to the header file of this hole. This can be used to link observations (e.g. assays, logging
        notes, or links to specific samples or external sites) with specific depths down hole.
        :param name: The (short) name to associate with this note, e.g., "Cu=0.5%" or "Cool stone!".
        :param value: The (longer) value to associate with this note, e.g., a full assay result or a link.
        :param depth_from: The start depth to associate with this note.
        :param depth_to: The end depth to associate with this note. If None (default) then start depth will be used.
        :param type: What type of annotation this is. Options are currently: "note" (default) or "link".
        :param group: A group name to associate this note with (helps organise data with lots of notes). Default is "notes".
        """
        if (depth_to is None):
            depth_to = depth_from
        top = np.min([b.start for b in self.getBoxes()])
        bottom = np.max([b.end for b in self.getBoxes()])
        assert (depth_to > top), "Error - note does not overlap with hole?"
        assert (depth_from < bottom), "Error - note does not overlap with hole?"
        self.header['%s_%s_%d_%d'%(type, group, depth_from*100, depth_to*100)] = "%s,%s"%(name, value)
    
    def delete_annotations(self):
        todel = []
        for k,v in self.header.items():
            if 'note' in k:
                todel.append(k)
        for k in todel:
            del self.header[k]
        io.save( os.path.splitext(self.getDirectory())[0] + ".hdr", self.header )
class Box( BaseCollection ):
    """
    Preprocessed and coregistered hyperspectral data from each of the sensors used in this Shed.
    """
    def __init__(self, name, root, *, ctype = "hycore.coreshed.Box", header=None):
        super().__init__(name, root, ctype=ctype, header=header)

        # defaults
        if not 'start' in self.header:
            self.start = 0
        if not 'end' in self.header:
            self.end = 1

    def holeName(self):
        """
        :return: The name of the drillhole (containing directory) that this box belongs to.
        """
        return os.path.splitext( os.path.basename( self.root ) )[0]

    def getFullName(self):
        """
        :return: Return the full name of this box as shed/hole/boxid.
        """
        box = self.name
        hole = os.path.splitext( os.path.basename( self.root ) )[0]
        shed = os.path.splitext( os.path.basename( os.path.dirname( self.root ) ) )[0]
        return "%s/%s/%s" % (shed,hole,box)

    def length(self):
        if (self.end is None) or (self.start is None):
            return 0 # no defined depths
        return self.end - self.start

    def getSensors(self):
        """
        Return a list of sensors included in this Box.
        """
        # force us to load from disk every time (ensure synchronisation between threads)
        if hasattr(self, 'sensor_list'):
            del self.sensor_list
        if not os.path.exists( os.path.join( self.getDirectory(), 'sensor_list.npy' ) ):
            return np.array([])
        else:
            return list( set( self.sensor_list ) ) # will load the numpy array via HyCollection parent class

    def addHSI(self, sensor, image):
        """
        Adds a hyperspectral image to this Shed.
        :param sensor: The name of the sensor that captured this HSI image (e.g. 'fenix').
        :param image: A HyImage instance containing the hyperspectral data.
        """
        self.set(sensor, image)

        # update sensor list
        self.sensor_list = np.hstack([self.getSensors(), sensor])
        os.makedirs(self.getDirectory(), exist_ok = True )
        np.save( os.path.join( self.getDirectory(), 'sensor_list.npy' ), self.sensor_list ) # save to disk immediately

    def getTemplate(self, *, method : str = 'sticks', axis : int = 0, flipx :  bool = False, flipy : bool = False,
                    thresh : float = 0.05, pad : int = 5, strict=False ):
        """
        Get an unwrapped template for this box, for creating mosaics (see hycore.templates).

        -------------------
        |    stick 1      |            ---------------------------
        |    stick 2      |       ==> | stick 1  stick 2  stick 3 |   if axis = 0
        |    stick 3      |            ---------------------------
        -------------------

        or

         -------------------
        |    stick 1      |            ----------
        |    stick 2      |       ==> | stick 1 |
        |                 |           | stick 2 |  if axis = 1
        |                 |           | stick 3 |
        |    stick 3      |            ----------
        -------------------

        :param method: The unwrapping method to use. Default is 'sticks' (unwrap core-sticks along the x-axis), although 'blocks' is also possible
                        (label_blocks will be used instead of label_sticks, treating every contiguous chunk in the mask as a different piece of core).
        :param axis: The axis to stack the unwrapped segments along.
        :param flipx: True if the sticks should be ordered from right-to-left.
        :param flipy: True if the sticks should be ordered from bottom-to-top.
        :param thresh: The threshold used to define breaks in the core (see get_breaks). Default is 20% of the max count.
        :param pad: Number of pixels to include between sticks and on the edge of the image.
        :param strict: If False (default), None will be returned if e.g. an appropriate mask is not defined for this box. If True, an error will be thrown.
        :return: A hycore.Template instance containing the template for this box.
        """

        # get template
        from hycore.templates import Template, unwrap_tray   # import here as otherwise we get a circular import error

        # check mask exists and is appropriate
        if not hasattr(self, 'mask'):
            if strict:
                assert False, "Error, box %s has no defined mask" % self.getFullName()
            else:
                print("Warning, box %s has no defined mask" % self.getFullName())
                return None
        if (self.mask.data == 0).all(): # blank mask
            if strict:
                assert False, "Error, mask for box %s is completely empty." % self.getFullName()
            else:
                print("Warning, mask for box %s is completely empty." % self.getFullName())
                return None

        # unwrap
        ixx = unwrap_tray( self.mask, method=method, axis=axis, thresh=thresh, flipx=flipx, flipy=flipy, pad=pad,
                           from_depth=self.start, to_depth=self.end )
        depths=ixx.header.get_list('depths')
        ticks=ixx.header.get_list('ticks')
        return Template( self.getDirectory(), ixx, from_depth=self.start, to_depth=self.end, depths=depths, depth_ticks=ticks)

    def quantize(self, sensors=None, save=True, mask=True, **kwargs ):
        """
        Update quantised representations of the specified sensors. These store lossy-compressed hyperspectral data
        by classifying each sensor into n spectral classes and storing only a minimum, median and maximum spectra for these
        (along side the classification image). These can be useful for e.g., running algorithms rapidly (as they need to
        process fewer spectra), doing target selection or data-driven sampling, or exporting data for visualisation (see
        the *hywiz* package).

        :param sensors: A list of sensors to quantise. If None (default) then self.getSensors() will be used.
        :param save: If True (default), the resulting index classification and spectral library objects will be saved in a box.spectra HyCollection.
        :param mask: If True, and a mask exists in this box, then areas outside of the masked area will be excluded during quantisation. Default is True.
        :keywords: all keywords are passed to `hylite.HyData.getQuantized( ... )`.
        :return: index, library = a list of classification index images for each sensor, and a list of libraries containing spectral info.
        """
        # get sensors
        if sensors is None:
            sensors = self.getSensors()
        elif isinstance(sensors, str):
            sensors = [sensors]

        if mask:
            if hasattr(self, 'mask'):
                kwargs['mask'] = self.mask.data[...,0] != 0
                if (not kwargs['mask'].any() ):
                    del kwargs['mask'] # get rid of mask if all False
                    
        index = []
        library = []
        for s in sensors:
            if hasattr(self, s):
                if (self.get(s).band_count() > 4): # only bother with multi or hyper spec data
                    ix,lb = self.get(s).getQuantized( **kwargs )
                    index.append(ix)
                    library.append(lb)
                    if save:
                        O = self.addSub('spectra')
                        O.set(s+"_idx", ix )
                        O.set(s+"_lib", lb )
                        O.save()
                        O.free()
        self.free()
        return index, library

    def unquantize(self, sensors=None, save=False):
        """
        Unpack quantized spectral data into hypercubes for the specified sensors.
        :param sensors: A list of sensors to quantise. If None (default) then self.getSensors() will be used.
        :param save: If true, this box will be saved after running this operation. Default is False.
        """
        # get sensors
        if sensors is None:
            sensors = self.getSensors()
        elif isinstance(sensors, str):
            sensors = [sensors]

        for s in sensors:
            # get index and library
            assert hasattr(self, 'spectra'), "Please generate quanta first using the `Box.quantize(...)` function."

            idx = self.spectra.get("%s_idx"%s)
            lib = self.spectra.get("%s_lib"%s)
            if isinstance(lib, hylite.HyImage):
                # convert from PNG format
                data = np.transpose( lib.data, (0,2,1) )
                data = data.astype(np.float32) / 255
                lib = hylite.HyLibrary( data, wav=lib.get_wavelengths() )

            # get mask
            self.mask = hylite.HyImage( (idx.data[...,0] != 0).astype(int) )

            # unpack sensor
            self.set( s, hylite.HyData.fromQuanta(idx,lib) )
        if save:
            self.save()
            self.free()

    def updateVis(self, sensors=None, qaqc=True, thumb=0 ):
        """
        Update the visualisation image(s) for this box, following the band mapping defined in the coreshed.ternary
        dictionary. Sensors not in this dictionary will be rendered as greyscale images using only their first band.

        :param sensor: The sensor name (str or list) to visualise. If None (default) then visualisations for all sensors
                       will be updated.
        :param qaqc: True if QAQC plots should also be created. Default is True.
        :param thumb: Either (1) an integer index in the sensors array to use as the thumbnail (default is 0), (2)
                      a string sensor name (a .png file of this name must exist!), or (3) None to disable.
        """

        if isinstance(sensors, str):
            sensors = [sensors]
        elif sensors is None:
            sensors = self.getSensors()


        # be quite forgiving with wavelength selection
        bst = hylite.band_select_threshold
        hylite.band_select_threshold = 100.

        for s in sensors:
            # get image
            free=False
            if not self.loaded(s):
                free = True
            source_img = self.get(s)

            # generate QAQC plots
            if qaqc:
                noise = hylite.HyImage(
                    np.nanmean(np.abs(np.diff(source_img.data, axis=-1)) / source_img.data[..., 1:], axis=-1)[..., None])
                sat = hylite.HyImage(np.mean(source_img.data, axis=-1)[..., None].astype(np.float32))
                oversat = np.isinf(noise.data)  # oversaturated pixels result in div 0
                noise.data[oversat] = np.nan  # replace these with nans
                noise.quick_plot(0, vmin=0., vmax=99, cmap='Spectral_r',
                                 path=os.path.join(self.getDirectory(), 'QAQC_%s_noise.png'%s) )
                plt.close() # prevent plots being stored / memory leak
                sat.data[oversat] = np.nan
                sat.quick_plot(0, vmin=0., vmax=100, cmap='Spectral_r',
                               path=os.path.join(self.getDirectory(), 'QAQC_%s_saturation.png' % s))
                plt.close()  # prevent plots being stored / memory leak

            # get bands
            bidx = [ source_img.get_band_index(b) for b in ternary.get(s, (0, 0, 0) ) ]
            v = hylite.HyImage( source_img.data[..., bidx].astype(np.float32) )

            # normalise
            v.percent_clip( minv=ternary_clip[0], maxv=ternary_clip[1], per_band=False ) # percent clip
            v.data = (v.data * 255).astype(np.uint8) # convert to 0-255 uint8 values

            # update header info
            v.header['file type'] = 'png'
            v.header['desc'] = 'Ternary preview for data from %s' % s
            v.header['bands'] = bidx
            v.set_wavelengths( np.array( ternary.get(s, (0,1,2) ) ) )
            v.set_band_names( ['band_%d'%d for d in bidx])

            # store
            io.save( os.path.join(self.getDirectory(), s + ".png"), v )

            if free: # free the image we loaded
                self.free_attr(s)

        # sort out thumbnails
        if thumb is not None:
            if isinstance(thumb, int):
                thumb = sensors[thumb]
            if ".png" not in thumb:
                thumb = thumb + ".png"

            # load source image for thumbnail
            thumb = self.get(thumb)
            self.free()

            # downsample and save
            ss = int( max(thumb.xdim(), thumb.ydim() ) / 128 )
            thumb.data = thumb.data[::ss,::ss,:]
            io.save( os.path.join(self.getDirectory(), "thumb.png"), thumb )

        hylite.band_select_threshold = bst # stop being so forgiving

