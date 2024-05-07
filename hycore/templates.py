"""
Functions for creating and arranging template files to create various types of mosiacs.
"""
import hylite
from hylite import io
import numpy as np
import os
from pathlib import Path
from collections.abc import MutableMapping

import hycore

# the below code avoids occasional errors when running large templates
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

################################################################
## Labelling functions. These identify or flag parts of a core
## box or sample that are of interest during mosaicing
################################################################
def get_bounds(mask: hylite.HyImage, pad: int = 0):
    """
    Get the bounds of the foreground area in the given mask.

    Args:
     - mask = A HyImage instance containing the foreground mask in the first band (background pixels flagged as 0 or False).
     - pad = Number of pixels padding to add to the masked area (N.B. this will not excede the image dimensions though).

    Returns:
     - xmin,xmax,ymin,ymax = The bounding box of the foreground pixels.
    """
    if isinstance(mask, hylite.HyImage):
        mask = mask.data[..., 0]
    else:
        mask = mask.squeeze()

    xmin = np.argmax(mask.any(axis=1))
    xmax = mask.shape[0] - np.argmax(mask.any(axis=1)[::-1])
    ymin = np.argmax(mask.any(axis=0))
    ymax = mask.shape[1] - np.argmax(mask.any(axis=0)[::-1])

    if pad > 0:
        xmin = max(0, xmin - pad)
        xmax = min(mask.shape[0], xmax + pad)
        ymin = max(0, ymin - pad)
        ymax = min(mask.shape[1], ymax + pad)

    return int(xmin), int(xmax), int(ymin), int(ymax)

def get_breaks(mask: hylite.HyImage, axis: int = 0, thresh: float = 0.2):
    """
    Identify breaks in the foreground mask as local minima after summing in the specified axis.

    Args:
     - axis = The axis along which to sum the mask before identifying minima.
     - thresh = the threshold used to define a "break", as a fraction of the maximum count (if a float is passed), or a specific value (if an int is past).
    """
    c = np.sum(mask.data[..., 0], axis=axis)
    if isinstance(thresh, float):
        thresh = np.max(c) * thresh

    breaks = np.argwhere(np.diff((c > thresh).flatten(), axis=0)).flatten()
    if len(breaks) > 2:
        breaks = 0.5 * (breaks[2:][::2] + breaks[1:-1][::2])
        return breaks.astype(int)
    else:
        return []  # no breaks

def label_sticks(mask: hylite.HyImage, axis: int = 0, thresh=0.2):
    """
    Identify and label sticks of core arranged in a box as follows:

     -------------------
    |    stick 1      |
    |    stick 2      |
    |    stick 3      |
    -------------------

    :param mask: A HyImage or numpy array containing 0s for all background and box pixels.
    :param axis: The long axis of each sticks. Default is 0 (x-axis).
    :param thresh: The threshold used to define breaks in the core (see get_breaks).
    :return: An updated mask with non-background pixels labelled according to their corresponding position
             in the core tray (from top to bottom if axis=0).
    """

    # get bounds and breaks
    xmin, xmax, ymin, ymax = get_bounds(mask)
    breaks = get_breaks(mask, axis=axis, thresh=thresh)

    # populate sticks
    idx = np.zeros((mask.xdim(), mask.ydim()))
    if axis == 0:
        steps = np.hstack([ymin, breaks, ymax])

        # build stick template from each step
        for n, (i0, i1) in enumerate(zip(steps[:-1], steps[1:])):
            idx[:, int(i0):int(i1)] = n + 1

    elif axis == 1:
        steps = np.hstack([xmin, breaks, xmax])
        for n, (i0, i1) in enumerate(zip(steps[:-1], steps[1:])):
            idx[int(i0):int(i1), :] = n + 1

    else:
        assert False, "Error - axis should be 0 or 1, not %s" % axis

    # intersect with mask
    idx[mask.data[..., 0] == 0] = 0

    return hylite.HyImage(idx)

def label_blocks(mask: hylite.HyImage):
    """
    Identify and label contiguous blocks. Useful for e.g. extracting samples or scanned thick-section blocks.
    :param mask: A HyImage or numpy array containing 0s for all background and box pixels.
    :return: An updated mask with non-background pixels labelled according to the contiguous block they belong to.
    """

    from skimage.measure import label
    return hylite.HyImage(label(mask.data[..., 0]))

################################################################
## Unwrap functions: These construct HyImage instances containing
## pixel mappings, that can be used to construct Templates
################################################################
def unwrap_bounds(mask: hylite.HyImage, *, pad: int = 1):
    """
    Construct and index template containing a mapping that just clips data to the mask (with the specified padding)

    :param mask: The mask that defines the clipping operation.
    :param pad: Any padding (in pixels) to apply to this. Default is 1.
    :return: A HyImage instance with a clipped shape and containing the x,y coordinates of the source pixels (for creating a template).
    """

    # get bounds and build indices
    xmn, xmx, ymn, ymx = get_bounds(mask, pad=pad)
    yy, xx = np.meshgrid(np.arange(mask.ydim()), np.arange(mask.xdim()))  # build coordinate arrays
    xy = np.dstack([xx, yy])
    xy[mask.data[..., 0] == 0, :] = -1
    idx = xy[xmn:xmx, ymn:ymx]
    return hylite.HyImage(idx)


def unwrap_tray(mask: hylite.HyImage, method='sticks', axis=0,
                flipx=False, flipy=False,
                thresh: float = 0.2,
                pad: int = 5, from_depth=0, to_depth=1):
    """
    Create a template that splits a core tray into individual "sticks"
    and then lays them end to end:

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


    :param mask: A HyImage instance containing 0s for all background and box pixels.
    :param method: The unwrapping method to use. Default is 'sticks' (see above), although 'blocks' is also possible
                    (label_blocks will be used instead of label_sticks).
    :param axis: The axis to stack the unwrapped segments along.
    :param flipx: True if the sticks should be ordered from right-to-left.
    :param flipy: True if the sticks should be ordered from bottom-to-top.
    :param thresh: The threshold used to define breaks in the core (see get_breaks). Default is 20% of the max count.
    :param pad: Number of pixels to include between sticks and on the edge of the image.
    :param from_depth: The depth of the start of this core box, for creating depth ticks. Default is 0. Tick positions and depth values will be stored in the resulting image's header.
    :param to_depth: The depth of the end of this core box, for creating depth ticks. Default is 1. Tick positions and depth values will be stored in the resulting image's header.
    :return: A HyImage instance containing the x,y coordinates of the unwrapped sticks.
    """
    if 'sticks' in method.lower():
        sticks = label_sticks(mask, axis=0, thresh=thresh)
    else:
        sticks = label_blocks(mask)

    yy, xx = np.meshgrid(np.arange(mask.ydim()), np.arange(mask.xdim()))  # build coordinate arrays
    xy = np.dstack([xx, yy])

    # extract chunks
    chunks = []
    for i in np.arange(1, np.max(sticks.data) + 1):
        msk = sticks.data[..., 0] == i  # get this segment
        xmn, xmx, ymn, ymx = get_bounds(msk, pad=pad)  # find its bounds
        idx = xy[xmn:xmx, ymn:ymx, :]  # get indices
        idx[~msk[xmn:xmx, ymn:ymx]] = -1  # also transfer background pixels
        chunks.append(xy[xmn:xmx, ymn:ymx, :])  # and store

    # stack chunks
    if axis == 0:
        xdim = np.sum([c.shape[0] for c in chunks])
        ydim = np.max([c.shape[1] for c in chunks])
    else:
        xdim = np.max([c.shape[0] for c in chunks])
        ydim = np.sum([c.shape[1] for c in chunks])

    idx = np.full((xdim, ydim, 2), -1, dtype=int)
    _o = 0
    ticks = []  # also store depth markers (ticks)
    depths = []  # and corresponding hole depths
    if flipy:
        chunks = chunks[::-1] # loop through chunks from bottom to top
    for i, c in enumerate(chunks):
        if flipx:
            c = c[::-1, :] # core runs right to left
        if axis == 0:
            idx[_o:(_o + c.shape[0]), 0:c.shape[1], :] = c
            ticks.append(_o)
            _o += c.shape[0]
        else:
            idx[0:c.shape[0], _o:(_o + c.shape[1]), :] = c
            ticks.append(_o + int(c.shape[1] / 2))
            _o += c.shape[1]
        depths.append(round(from_depth + i * (to_depth - from_depth) / len(chunks), 2))

    out = hylite.HyImage(idx)
    out.header['depths'] = depths
    out.header['ticks'] = ticks
    out.header['tickAxis'] = axis
    return out

################################################################
## Template factory
## These functions create templates for sets of boxes.
################################################################
def buildStack(boxes: hycore.Box, *, pad=1, axis=0, transpose=False):
    """
    Build a simple template that stacks data horizontally or vertically

    :param boxes: A list of Box objects to stack.
    :param pad: Padding to add between boxes. Default is 1.
    :param axis: 0 to stack horizontally, 1 to stack vertically.
    :param transpose: If True, templates are transposed before stacking.
    :return: A template object containing the stacked indices.
    """

    # get shed directory
    templates = []
    for b in boxes:
        try:
            mask = b.mask
        except:
            assert False, "Error - box %s must have a mask defined for it to be included in the stack" % b.name

        # get clip index and construct template
        iClip = unwrap_bounds(mask, pad=pad)
        if transpose:
            iClip.data = np.transpose(iClip.data, (1, 0, 2))

        templates.append(Template(b.getDirectory(), iClip,
                                  from_depth = b.start, to_depth = b.end,
                                  groups=[b.name], group_ticks=[iClip.xdim() / 2 ]))

    # do stack
    return Template.stack(templates, axis=axis )

################################################################
## Templates and Canvas classes encapsulate
## mosaicing transformations and can be used to push data around
## to derive mosaic images.
################################################################

def compositeStack(template, images : list, bands : list = [(0, 1, 2)], *, axis=1, vmin=2, vmax=98, **kwargs):
    """
    Pass multiple images through the provided template and then stack the results along the specified axis.

    :param template: The Template object to use to define mapping between output images and input images.
    :param images: A list of image names to resolve. If a single image is passed then this function reduces to be equivalent
                    to Template.apply( ... ).
    :param axis: The axis to stack the resulting output images along. Default is 1 (stack vertically along the y-axis).
    :param bands: A list of tuples defining the bands to be visualised for each image listed in `images`. These tuples
                  must have matching lengths!
    :param vmin: Percentile clip to apply separately to each dataset before doing stacking. Set as None to disable.
    :param vmax: Percentile clip to apply separately to each dataset before doing stacking. Set as None to disable.
    :param kwargs: All keyword arguments are passed to Template.apply
    :return: A composited and stacked image.
    """

    # apply template to all input images
    I = []
    if not isinstance(bands, list): # allow people to pass e.g., (0,1,2) bands and have this extended
        bands = [bands] * len(images) # for each image.
    for i, b in zip(images, bands):
        I.append( template.apply(i, b, **kwargs) )
        if (vmin is not None):
            I[-1].percent_clip( vmin, vmax )

    # stack results
    out = I[0]
    h = out.data.shape[axis]
    out.data = np.concatenate([i.data for i in I], axis=axis)

    # add some metadata that can be useful for later plotting
    out.header['images'] = images
    out.header['bands'] = bands
    out.header['image_ticks'] = [i*h + 0.5 * h for i in range(len(images))]
    out.set_wavelengths(None) # remove wavelength info as this is invalid

    # return
    return out

class Template(object):
    """
    A special type of image dataset that stores drillcore, box and pixel IDs such that data from
    multiple sources can be re-arranged and combined into mosaick images.
    """

    def __init__(self, boxes: list, index: np.ndarray, from_depth : float = None, to_depth : float = None,
                 groups=None, group_ticks=None, depths=None, depth_ticks=None, depth_axis=0):
        """
        Create a new Template instance.
        :param boxes: A list of paths for each box directory that is included in this template.
        :param index: An index array defining where data for index[x,y] should be sourced from. This should have four
                        bands specifying the: holeID (index in holes list), boxID (index in boxes list), xcoord (in source
                        image) and ycoord (in source image).
        :param from_depth: The top (smallest depth) of this template.
        :param to_depth: The bottom (largest depth) of this template.
        :param groups: Group names (e.g., boreholes) in this template.
        :param group_ticks: Pixel coordinates of ticks for these groups.
        :param depths: Depth values for ticks in the depth direction.
        :param depth_ticks: Pixel coordinates of the ticks corresponding to these depth values.
        :param depth_axis: Axis of this template that depth ticks correspond to.
        """

        # check data types
        if isinstance(index, hylite.HyImage):
            index = index.data
        if isinstance(boxes, str) or isinstance(boxes, Path):
            boxes = [boxes]  # wrap in a list

        # get root path and express everything as relative to that
        root = os.path.commonpath(boxes)
        if root == boxes[0]: # only one box (or all the same)
            root = os.path.dirname(boxes[0])
            self.boxes = [os.path.basename(b) for b in boxes]
        else:
            self.boxes = [os.path.relpath(b, root) for b in boxes]

        for b in self.boxes:
            assert os.path.exists(os.path.join(root, b)), "Error box %s does not exist." % os.path.join(root, b)

        # check dimensionality of index
        if index.shape[-1] == 2:
            index = np.dstack([np.zeros((index.shape[0], index.shape[1]), dtype=int), index])

        assert index.shape[-1] == 3, "Error - index must have 3 bands (box_index, xcoord, ycoord)"

        self.root = str(root)
        self.index = index.astype(int)

        if (from_depth is not None) and (to_depth is not None):
            self.from_depth = min( from_depth, to_depth)
            self.to_depth = max( from_depth, to_depth )
            self.center_depth = 0.5*(from_depth + to_depth)
        else:
            self.center_depth = None
            self.from_depth = None
            self.to_depth = None

        self.groups = None
        self.group_ticks = None
        self.depths = None
        self.depth_ticks = None
        self.depth_axis = depth_axis

        if groups is not None:
            self.groups = np.array(groups)
        if group_ticks is not None:
            self.group_ticks = np.array(group_ticks)
        if depths is not None:
            self.depths = np.array(depths)
        if depth_ticks is not None:
            self.depth_ticks = np.array(depth_ticks)

    def apply(self, imageName, bands=None, strict=False, xstep : int = 1, ystep : int = 1, scale : float = 1,
                    outline=None):
        """
        Apply this template to the specified dataset in each box.

        :param imageName: The name of the image file to extract data from in each box, e.g., 'FENIX'.
        :param bands: The bands to export from the source image. Default is None (export all bands). See HyData.export_bands for possible formats.
        :param strict: If False (default), skip files that could not be found.
        :param xstep: Step to use in the x-direction. Useful for skipping pixels in the source image when building large mosaics.
        :param ystep: Step to use in the y-direction. Useful for skipping pixels in the source image when building large mosaics.
        :param scale: Scale factor to apply to pixel coordinates in this mosaic. Used for applying to images that are different resolutions.
        :param outline: Tuple containing the colour used to outline masked areas, or None to disable.
        :return: A HyImage instance containing the mosaic populated with the requested bands.
        """
        out = None
        for i, box in enumerate(self.boxes):

            # get this box
            path = os.path.join( self.root, box )
            if strict:
                assert os.path.exists(path), "Error - could not load data from %d"

            # load the required data
            try:
                try:
                    if '.' in imageName:
                        data = io.load(os.path.join(path, imageName) ) # extension is provided
                    else:
                        data = io.load(path).get(imageName) # look in box directory
                except (AttributeError, AssertionError) as e:
                    try:
                        if '.' in imageName:
                            data = io.load(os.path.join(path, 'results.hyc/%s'%imageName))  # extension is provided
                        else:
                            data = io.load(path).results.get(imageName) # look in results directory
                    except:
                        if strict:
                            assert False, "Error - could not find data %s in directory %s" % (imageName, path)
                        else:
                            continue

                data.decompress()
            except (AttributeError, AssertionError) as e:
                if strict:
                    assert False, "Error - could not find data %s in directory %s" % (imageName, path )
                else:
                    continue

            # get index and subsample as needed
            index = self.index[::xstep, ::ystep]
            if scale != 1:
                index = (index * scale).astype(int) # scale coordinates
                index[...,0] = self.index[::xstep, ::ystep, 0] # don't scale box IDs

            # create output array
            if bands is not None:
                data = data.export_bands(bands)
            if out is None:
                # initialise output array now we know how many bands we're dealing with
                out = np.zeros( (index.shape[0], index.shape[1], data.band_count() ), dtype=data.data.dtype )

            # copy data as defined in index array
            mask = (index[..., 0] == i) & (index[..., -1] != -1 )
            if mask.any():
                out[ mask, : ] = data.data[ index[mask, 1], index[mask, 2], : ]
        if len( data.get_wavelengths() ) != data.band_count():
            data.set_wavelengths(np.arange(data.band_count()))

        # return a hyimage
        out = hylite.HyImage( out, wav=data.get_wavelengths() )
        if data.has_band_names():
            if len(data.get_band_names()) == out.band_count(): # sometimes this is not the case if we load a PNG with a header from a .dat file!
                out.set_band_names(data.get_band_names())

        if (self.groups is not None) and (len(self.groups) > 0):
            out.header['groups'] = self.groups
        if (self.group_ticks is not None) and (len(self.group_ticks) > 0):
            out.header['group ticks'] = self.group_ticks

        if outline is not None:
            self.add_outlines(out, color=outline, xx = xstep, yy = ystep )
        return out

    def quick_plot(self, band=0, rot=False, xx=1, yy=1, interval=5, **kwds):
        """
        Quickly plot this template for QAQC.
        :param band: The band(s) to plot. Default is 0 (plot only box ID).
        :param rot: True if the template should be rotated 90 degrees before plotting.
        :param xx: subsampling in the x-direction (useful for large templates!). Default is 1 (no subsampling).
        :param yy: subsampling in the y-direction (useful for large templates!). Default is 1 (no subsampling).
        :param interval: Interval between depth ticks to add to plot.
        :keywords: Keywords are passed to hylite.HyImage.quick_plot( ... ).
        :return: fig,ax from the matplotlib figure created.
        """
        img = self.toImage()
        img.data = img.data[ ::xx, ::yy, : ]
        if rot:
            img.rot90()
        fig, ax = img.quick_plot(band, tscale=True, **kwds )
        self.add_ticks(ax, rot=rot, xx=xx, yy=yy, interval=interval)
        return fig, ax

    def add_ticks(self, ax, interval=5, *, depth_ticks: bool = True, group_ticks: bool = True, rot: bool = False,
                  xx: int = 1, yy: int = 1, angle: float = 45):
        """
        Add depth and or group ticks (as stored in this template) to a matplotlib plot.

        :param ax: The matplotlib axes object to set x- and y- ticks / labels too.
        :param interval: The interval (in m) between depth ticks. Default is 5 m.
        :param depth_ticks:  True (default) if depth ticks should be plotted.
        :param group_ticks: True (default) if group ticks should be plotted.
        :param rot: If True, the x- and y- axes are flipped (e.g. if image was rotated relative to this template before plotting).
        :param xx: subsampling in the x-direction (useful for large templates!). Default is 1 (no subsampling).
        :param yy: subsampling in the y-direction (useful for large templates!). Default is 1 (no subsampling).
        :param angle: rotation used for the x-ticks. Default is 45 degrees.
        """
        a = self.depth_axis
        if rot:
            a = int(1 - a)
            _xx = xx
            xx = yy
            yy = _xx

        # get depth and group ticks
        if depth_ticks:
            zt, zz = self.get_depth_ticks( interval )
        if group_ticks:
            gt,gg = self.get_group_ticks()

        if a == 0:
            if depth_ticks:
                ax.set_xticks(zt / xx )
                ax.set_xticklabels( ["%.1f" % z for z in zz], rotation=angle )
                #ax.tick_params('x', labelrotation=angle )
            if group_ticks and self.group_ticks is not None:
                ax.set_yticks(gt / yy )
                ax.set_yticklabels( ["%s" % g for g in gg] )
        else:
            if depth_ticks:
                ax.set_yticks(zt / xx )
                ax.set_yticklabels(["%.1f" % z for z in zz])
            if group_ticks:
                ax.set_xticks(gt / yy)
                ax.set_xticklabels(["%s" % g for g in gg], rotation=angle)
                #ax.tick_params('x', labelrotation=angle)

    def get_group_ticks(self):
        """
        :return: The position and label of group ticks defined in this template, or [], [] if None are defined.
        """
        if (self.groups is not None) and (self.group_ticks is not None):
            return self.group_ticks, self.groups
        else:
            return np.array([]), np.array([])

    def get_depth_ticks(self, interval=1.0):
        """
        Get evenly spaced depth ticks for pretty plotting.
        :param interval: The desired spacing between depth ticks
        :return: Depth tick positions and values. If depth_ticks and depths are not defined, this will return empty lists.
        """
        if (self.from_depth is None) or (self.to_depth is None) or (self.depth_ticks is None) or (self.depths is None):
            return np.array([]), np.array([])
        else:
            zz = np.arange(self.from_depth - self.from_depth % interval,
                           self.to_depth + interval - self.to_depth % interval, interval )[1:]

            tt = np.interp( zz, self.depths, self.depth_ticks)

            return tt, zz

    def add_outlines(self, image, color=0.4, mode='thick', xx: int = 1, yy: int = 1):
        """
        Add outlines from this template to the specified image.

        :param image: a HyImage instance to add colours too. Note that this will be updated in-place.
        :param color: a float or tuple containing the values of the colour to apply.
        :param mode: outline mode. Options are ‘thick’, ‘inner’, ‘outer’, ‘subpixel’ (see skimage.segmentation.mark_boundaries for details).
        """
        dtype = image.data.dtype  # store this for later

        # get mask to outline
        mask = self.index[::xx, ::yy, 1] != -1

        # sort out colour
        if isinstance(color, float) or isinstance(color, int):
            color = tuple([color for i in range(image.band_count())])
        assert len(color) == image.band_count(), "Error - colour must have same number of bands as image. %d != %d" % (
        len(color), image.band_count())
        if (np.array(color) > 1).any():
            color = np.array(color) / 255.

        # mark boundaries using scikit-image
        from skimage.segmentation import mark_boundaries
        image.data = mark_boundaries(image.data, mask, color=color, mode=mode)

        if (dtype == np.uint8):
            image.data = (image.data * 255)  # scikit image transforms our data to 0 - 1 range...

    def toImage(self):
        """
        Convert this Template object to a HyImage instance with the relevant additional hole and box lists stored
        in the header file. This can be saved and then later converted back to a Template using fromImage( ... ).
        :return: A HyImage representation of this template.
        """
        image = hylite.HyImage(self.index)
        image.header['root'] = self.root
        image.header['boxes'] = self.boxes
        if self.from_depth is not None:
            image.header['from_depth'] = self.from_depth
        if self.to_depth is not None:
            image.header['to_depth'] = self.to_depth
        if self.center_depth is not None:
            image.header['center_depth'] = self.center_depth
        if self.groups is not None:
            image.header['groups'] = self.groups
        if self.group_ticks is not None:
            image.header['group_ticks'] = self.group_ticks
        if self.depths is not None:
            image.header['depths'] = self.depths
        if self.depth_ticks is not None:
            image.header['depth_ticks'] = self.depth_ticks
        if self.depth_axis is not None:
            image.header['depth_axis'] = self.depth_axis

        return image

    @classmethod
    def fromImage(cls, image):
        """
        Convert a HyImage with the relevant header information to a Template instance. Useful for IO.
        :param image: The HyImage instance containing the template mapping and relevant header metadata
                        (lists of hole and box names).
        :return:
        """
        assert 'root' in image.header, 'Error - image must have a "root" key in its header'
        assert 'boxes' in image.header, 'Error - image must have a "boxes" key in its header'
        assert image.band_count() == 3, 'Error - image must have four bands [holeID, boxID, xidx, yidx]'
        root = image.header['root']
        boxes = image.header.get_list('boxes')
        from_depth = None
        to_depth = None
        groups = None
        group_ticks = None
        depths = None
        depth_ticks = None
        depth_axis = None
        if 'from_depth' in image.header:
            from_depth = float(image.header['from_depth'])
        if 'to_depth' in image.header:
            to_depth = float(image.header['to_depth'])
        if 'groups' in image.header:
            groups = image.header.get_list('groups')
        if 'group_ticks' in image.header:
            group_ticks = image.header.get_list('group_ticks')
        if 'depths' in image.header:
            depths = image.header.get_list('depths')
        if 'depth_ticks' in image.header:
            depth_ticks = image.header.get_list('depth_ticks')
        if 'depth_axis' in image.header:
            depth_axis = int(image.header['depth_axis'])
        return Template([os.path.join( root, b) for b in boxes], image.data, from_depth=from_depth, to_depth=to_depth,
                        groups=groups, group_ticks=group_ticks, depths=depths, depth_ticks=depth_ticks, depth_axis=depth_axis)

    def rot90(self):
        """
        Rotate this template by 90 degrees.
        """
        self.index = np.rot90(self.index, axes=(0, 1))
        self.depth_axis = int(1 - self.depth_axis)

    def crop(self, min_depth : float, max_depth : float, axis : int , offset : float = 0):
        """
        Crop this template to the specified depth range.

        :param min_depth: The minimum allowable depth.
        :param max_depth: The maximum allowable depth.
        :param axis: The axis along which depth is interpolated in this template. Should be 0 (x-axis is depth axis) or 1 (y-axis is depth axis).
        :param offset: A depth to subtract from min_depth and max_depth prior to cropping.
        :return: A copy of this template, cropped to the specific range, or None if no overlap exists.
        """
        # check there is overlap
        if (self.from_depth is None) or (self.to_depth is None):
            assert False, "Error - template has no depth information."

        # interpolate depth
        zz = np.linspace( self.from_depth, self.to_depth, self.index.shape[axis] ) - offset
        mask = (zz >= min_depth) & (zz <= max_depth)
        if not mask.any():
            return None # no overlap

        if axis == 0:
            ix = self.index[mask, :, : ]
        else:
            ix = self.index[:, mask, : ]

        # print( min_depth, max_depth, self.from_depth, self.to_depth, np.min(zz[mask]), np.max(zz[mask]) )
        return Template( [os.path.join(self.root, b) for b in self.boxes], ix,
                         from_depth = np.min(zz[mask]),
                         to_depth = np.max(zz[mask]) ) # return cropped template

    @classmethod
    def stack(cls, templates: list, xstep : int = 1, ystep : int = 1, axis=1):
        """
        Stack a list of templates along the specified axis (similar to np.vstack and np.hstack).

        :param templates: A list of template objects to stack.
        :param xstep: Step to use in the x-direction. Useful for skipping pixels in the source image when generating large mosaics.
        :param ystep: Step to use in the y-direction. Useful for skipping pixels in the source image when generating large mosaics.
        :param axis: The axis to stack along. Set as zero to stack in the x-direction and 1 to stack in the
                     y-direction.
        """

        # resolve all unique paths
        paths = set()
        for t in templates:
            for b in t.boxes:
                paths.add(os.path.join(t.root, b))
                assert os.path.exists(os.path.join(t.root, b)), "Error - one or more template directories do not exist?"

        # get root (lowest common base) and express boxes as relative paths to this
        paths = list(paths)

        # initialise output
        if axis == 0:
            out = np.full((sum([t.index[::xstep, ::ystep, :].shape[0] for t in templates]),
                            max([t.index[::xstep, ::ystep, :].shape[1] for t in templates]), 3), -1)
        else:
            out = np.full((max([t.index[::xstep, ::ystep, :].shape[0] for t in templates]),
                            sum([t.index[::xstep, ::ystep, :].shape[1] for t in templates]), 3), -1)

        # loop through templates and stack
        p = 0
        groups = []
        group_ticks = []
        for i, t in enumerate(templates):
            # copy block of indices across
            if axis == 0:
                out[p:(p + t.index[::xstep, ::ystep, :].shape[0]),
                        0:t.index[::xstep, ::ystep, :].shape[1], :] = t.index[::xstep, ::ystep, :]
            else:
                out[0:t.index[::xstep, ::ystep, :].shape[0],
                p:(p + t.index[::xstep, ::ystep, :].shape[1]), :] = t.index[::xstep, ::ystep, :]

            # update box indices
            for j, b in enumerate(t.boxes):
                mask = np.full((out.shape[0], out.shape[1]), False)
                if axis == 0:
                    mask[p:(p + t.index[::xstep, ::ystep, :].shape[0]), 0:t.index[::xstep, ::ystep, :].shape[1]] = (t.index[::xstep, ::ystep, 0] == j)
                else:
                    mask[0:t.index[::xstep, ::ystep, :].shape[0], p:(p + t.index[::xstep, ::ystep, :].shape[1])] = (t.index[::xstep, ::ystep, 0] == j)
                out[mask, 0] = paths.index(os.path.join(t.root, b))

            # update groups and group ticks (these are useful for subsequent plotting)

            if t.groups is not None:
                groups += list(t.groups)
                if axis == 0:
                    group_ticks += list( np.array(t.group_ticks) / xstep + p )
                else:
                    group_ticks += list(np.array(t.group_ticks) / ystep + p)

            # update start point
            p += t.index[::xstep, ::ystep, :].shape[axis]

        # get span of depths
        from_depth = None
        to_depth = None
        if np.array([t.center_depth is not None for t in templates]).all():
            from_depth = np.min([t.from_depth for t in templates])
            to_depth = np.max([t.to_depth for t in templates])

        # generate depth ticks
        # i = 1 - axis # if axis is 1, we tick along axis = 0, if axis is 0, we tick along axis = 1
        ticks = [templates[0].index.shape[axis] / 2]
        depths = [templates[0].from_depth]
        for i, T in enumerate(templates[1:]):
            ticks.append(ticks[-1] + templates[i - 1].index.shape[axis] / 2 + T.index.shape[axis] / 2)
            depths.append(T.from_depth)

        # return new Template instance
        return Template(paths, out, from_depth = from_depth, to_depth = to_depth,
                        groups=groups, group_ticks=group_ticks,
                        depths=depths, depth_ticks=ticks, depth_axis=axis)

    def getDepths(self, res: float = 1e-3):
        """
        Return a 1D array of the depths corresponding to each pixel. Assumes a linear mapping
        between the templates from_depth and to_depth.
        :param res: The known resolution of the image data. If None, depths are simply stretched evenly between
                    the start and end of this template. If specified, the start_depth is used as an
                    anchor and the depth of pixels below this computed to match the resolution. This is important
                    to preserve true scale when core boxes contain gaps.
        :return: A 1D array containing depth information for each pixel in this template.
        """
        axis = self.depth_axis
        if res is None:
            return np.linspace(self.from_depth, self.to_depth, self.index.shape[axis])
        else:
            to_depth = self.from_depth + self.index.shape[axis] * res
            return np.linspace(self.from_depth, to_depth, self.index.shape[axis])

    def getGrid(self, grid=50, minor=True, labels=True, background=True, res : float = 1e-3):
        """
        Create a depth grid image to accompany HSI mosaics.

        :param grid: The grid step, in mm. Default is 50.
        :param minor: True if minor ticks (with half the spacing of the major ticks) should be plotted.
        :param labels: True if label text describing the meterage should be added.
        :param background: True if background outlines of the core blocks should be added.
        :param res: The known resolution of the image data. If None, depths are simply stretched evenly between
                    the start and end of this template. If specified, the start_depth is used as an
                    anchor and the depth of pixels below this computed to match the resolution. This is important
                    to preserve true scale when core boxes contain gaps.
        :return: A HyImage instance containing the grid image.
        """
        # import this here in case of problematic cv2 install
        import cv2

        # get background image showing core blocks
        img = np.zeros((self.index.shape[0], self.index.shape[1], 3), dtype=np.uint8)
        if background:
            img[:, :, 1] = img[:, :, 2] = 120 * (self.index[:, :, 2] > 1)

        # interpolate depth
        zz = self.getDepths(res=res)

        # add ticks
        ignore = set()
        for i, z in enumerate(zz):
            zi = int(z * 1000)

            # major ticks
            if zi not in ignore:
                if (zi % int(grid)) == 0:
                    # add tick
                    img[i, :, :] = 255
                    ignore.add(zi)

                    # add depth label
                    if labels:
                        l = "%.2f" % z
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        img = cv2.putText(img,
                                          l, (0, i - 3), font, 0.5, (255, 255, 255), 1, bottomLeftOrigin=False)
            # minor ticks
            if (zi not in ignore) and minor:
                if (int(z * 1000) % int(grid / 2)) == 0:
                    img[i, ::3, :] = 255
                    ignore.add(zi)

        return hylite.HyImage(img)

    def __lt__(self, other):
        """
        Do comparisons based on depth of centerpoint. Used for quickly sorting templates by depth.
        """
        if self.center_depth is None:
            assert False, 'Error - please define depth data for Template to use < functions.'
        if isinstance(other, Template):
            if other.center_depth is None:
                assert False, 'Error - please define depth data for Template to use < functions.'
            return self.center_depth < other.center_depth
        else:
            return other > self.center_depth # use operator from other class

    def __gt__(self, other):
        """
        Do comparisons based on depth of centerpoint. Used for quickly sorting templates by depth.
        """
        if self.center_depth is None:
            assert False, 'Error - please define depth data for Template to use > functions.'
        if isinstance(other, Template):
            if other.center_depth is None:
                assert False, 'Error - please define depth data for Template to use < functions.'
            return self.center_depth > other.center_depth
        else:
            return other < self.center_depth # use operator from other class


class Canvas(MutableMapping):
    """
    A utility class for creating collections of templates and combining them into potentially complex layouts. This
    stores groups of templates, which can then be sorted and arranged in various ways (e.g., arranging groups as
    columns and cropping to a specific depth range, with individual drillhole offsets).
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys


    def hpole(self, from_depth: float = None, to_depth: float = None, scaled=False, groups: list = None,
                    res: float = 1e-3, depth_offsets: dict = {}, pad: int = 5 ):
        """
        Construct a "horizontal pole" type template for visualising and corellating between one or more drillholes.
        This has a layout as follows:

                          -------------------------------------------------
        core (group) 1 - |  [xxxxxxxxxx] [xxxx]         [xxxxxxxxxxxxxxx]  |
        core (group) 2 - |  [xxxxx]       [xxxxxxxxxxxx]         [xxxxxx]  |
        core (group) 3 - |  [xxxxxxxx] [xxxxxxxxxxxxxxxxxx][xxxxxxxxxxxx]  |
                          -------------------------------------------------

        :param from_depth: The top depth of the template view area, or None to include all depths.
        :param to_depth: The lower depth of the template view area, or None to include all depths.
        :param scaled: If True, a constant scale will be used on the z-axis. If False (default), cores will be stacked vertically
                (with small gaps representing non-contiguous intervals).
        :param groups: Names of the groups to plot (in order!). If None (default) then all groups are plotted.
        :param res: Resolution of the imagery in meters (used when deriving vertical scale). Defaults to 1e-3 (1 mm).
        :param depth_offsets: A dictionary containing depth values to be subtracted from sub-templates with matching
                                group names. Useful for e.g., plotting boreholes relative to a marker horizon rather than
                                in absolute terms.
        :param pad: Padding for template stacking. Default is 5.
        :return: A single combined Template class in horizontal pole layout.
        """

        S, from_depth, to_depth, groups, paths = self._preprocessTemplates(depth_offsets, from_depth,
                                                                           groups, to_depth )

        # compute width of output image
        w = pad
        for g in groups:
            w += np.max([T.index.shape[1] for T in S[g.lower()]]) + pad

        if scaled:
            # compute image dimension in depth direction
            nz = int(np.abs(to_depth - from_depth) / res)

            # compute corresponding depths
            z = np.linspace(from_depth, to_depth, nz)
        else:
            # determine maximum size of stacked boxes, including gaps, and hence template dimensions
            z = []
            for g in groups:
                nz = 0 # this is the dimensions of our output in pixels
                for i, T in enumerate(S[g.lower()]):
                    if (i > 0) and (abs(T.from_depth - S[g.lower()][i-1].to_depth) > 0.5):
                        nz += 10*pad # add in gaps for non-contiguous cores
                        z.append( np.linspace(S[g.lower()][i-1].to_depth, T.from_depth, 10*pad ) )

                    nz += T.index.shape[0] + pad
                    z.append(np.linspace(T.from_depth, T.to_depth, T.index.shape[0]))
                    z.append([T.to_depth for i in range(pad)])
            z = np.hstack(z)

        assert len(z) == nz, "Error - %d depths and %d pixels. Should be the same." % (len(z), nz) # debugging

        # build index
        index = np.full((nz, w, 3), -1, dtype=int)
        tticks = []  # store tick positions in transverse direction (y-axis for hpole)

        # stack templates
        _y = pad
        for g in groups:
            g = g.lower()
            for T in S[g]:
                # find depth position of center and copy data across
                if len(T.boxes) > 1:
                    assert False, "Error, cannot use multi-box templates on a Canvas (yet)"
                else:
                    six = int(np.argmin(np.abs(z - T.from_depth)))  # start index in z array
                    eix = min(T.index.shape[0],
                              (index.shape[0] - six))  # end index in template (to allow for possible overflows)

                    # copy data!
                    bix = int(paths.index(os.path.join(T.root, T.boxes[0])))
                    index[six:(six + T.index.shape[0]), _y:(_y + T.index.shape[1]), 0] = bix  # set box index

                    index[six:(six + eix), _y:(_y + T.index.shape[1]), 1:] = T.index[0:eix, :,
                                                                             1:]  # copy pixel indices

            # step to the right
            h = int(np.max([T.index.shape[1] for T in S[g]]) + pad)
            tticks.append(int(_y + (h / 2)))
            _y += h

        out = Template(paths, index, from_depth, to_depth, depth_axis=0,
            groups = groups, group_ticks = tticks, depths = z, depth_ticks = np.arange(len(z)))

        # done!
        return out

    def vfence(self, from_depth: float = None, to_depth: float = None, scaled=False,
               groups: list = None, depth_offsets : dict = {}, pad: int = 5):
        """
        Construct a "horizontal fence" type template for visualising drillholes in a condensed way.
        This has a layout as follows:

            core 1       core 2         core 3
    1  - | ======== | | =========| | ========== |
         | ======== | | =========| | ========== |
    2  - | ======== | | ======   | | =====      |
         | ======== |     gap      | ========== |
    3  - | ======   | | =========| | ========== |
         | ======== | | =========| | ========== |

        :param from_depth: The top depth of the template view area, or None to include all depths.
        :param to_depth: The lower depth of the template view area, or None to include all depths.
        :param scaled: If True, a constant scale will be used on the z-axis. If False (default), cores will be stacked vertically
                        (with small gaps representing non-contiguous intervals).
        :param groups: Names of the groups to plot (in order!). If None (default) then all groups are plotted.
        :param res: Resolution of the imagery in meters (used when deriving vertical scale). Defaults to 1e-3 (1 mm).
        :param depth_offsets: A dictionary containing depth values to be subtracted from sub-templates with matching
                                group names. Useful for e.g., plotting boreholes relative to a marker horizon rather than
                                in absolute terms.
        :param pad: Padding for template stacking. Default is 5.
        :return: A single combined Template class in horizontal pole layout.
        """

        S, from_depth, to_depth, groups, paths = self._preprocessTemplates(depth_offsets, from_depth,
                                                                           groups, to_depth )

        # compute width used for each group and hence image width
        # also compute y-scale based maximum template height to depth covered ratio
        w = pad # width
        ys = np.inf # shared y-axis pixel to depth scale (meters per pixel)
        for g in groups:
            w = w + np.max([T.index.shape[0] for T in S[g.lower()]]) + pad
            for T in S[g.lower()]:
                ys = min( ys, abs(T.to_depth - T.from_depth) / T.index.shape[1] )

        # compute image dimension in depth direction
        if scaled:
            # determine depth-scale along y-axis (distance down hole per pixel)
            nz = int(np.abs(to_depth - from_depth) / ys)
            z = np.linspace(from_depth, to_depth, nz ) # depth per pixel array (kinda...)

            # build index
            index = np.full((w, nz + pad, 3), -1, dtype=int)

        else:
            # determine maximum height of stacked boxes, including gaps, and hence template dimensions
            heights = []
            for g in groups:
                h = 0
                for i, T in enumerate(S[g.lower()]):
                    h += T.index.shape[1] + pad
                    if (i > 0) and (abs(T.from_depth - S[g.lower()][i-1].to_depth) > 0.5):
                        h += T.index.shape[1] # add in gaps for non-contiguous cores
                heights.append(h)

            ymax = np.max( heights )

            # build index
            index = np.full((w, ymax + pad, 3), -1, dtype=int)

        tticks = []  # store group tick positions in transverse direction (x-axis)

        # stack templates
        _x = pad
        for g in groups: # loop through groups
            zticks = []  # store depth ticks in the down-hole direction (y-axis)
            zvals = []  # store corresponding depth values

            g = g.lower()
            six=0
            for i,T in enumerate(S[g]): # loop through templates in this group
                # find depth position of center and copy data across
                if len(T.boxes) > 1:
                    assert False, "Error, cannot use multi-box templates on a Canvas (yet)"
                else:
                    if scaled:
                        six = int(np.argmin(np.abs(z - T.from_depth)))  # start index in z array
                    else:
                        # add gaps for non-contiguous templates
                        if i > 0 and (abs(S[g][i - 1].to_depth - T.from_depth) > 0.5):
                            six += T.index.shape[1]  # add full-box sized gap

                    # copy data!
                    bix = int(paths.index(os.path.join(T.root, T.boxes[0])))
                    index[ _x:(_x + T.index.shape[0] ) , six:(six+T.index.shape[1]), 0 ] = bix
                    index[ _x:(_x + T.index.shape[0] ) , six:(six+T.index.shape[1]), 1:] = T.index[:, :, 1:]  # copy pixel indices

                    # store depth ticks
                    zticks.append(six)
                    zvals.append(T.from_depth)

                    if not scaled:
                        six += T.index.shape[1]+pad # increment position

            # step to the right
            w = int(np.max([T.index.shape[0] for T in S[g]]) + pad) # compute max width of core blocks in this group
            tticks.append(int(_x + (w / 2))) # store group ticks
            _x += w # step to the right

        zticks.append(index.shape[1])
        zvals.append(T.to_depth) # add tick at bottom of final template / box

        if scaled and len(groups) > 1:
            zvals = None
            depth_ticks = None # these are not defined if more than one hole is present
        else:
            # interpolate zvals to get a depth value for each pixel
            zvals = np.interp(np.arange(0,index.shape[1]), zticks, zvals )
            zticks = np.arange(index.shape[1])
        out = Template(paths, index, from_depth, to_depth, depth_axis=1,
                       groups = groups, group_ticks = tticks, depths = zvals, depth_ticks = zticks )

        # done!
        return out


    def hfence(self, *args):
        """
        Construct a "horizontal fence" type template for visualising boreholes in a condensed way. This
        is identical to the vfence(...) layout, but rotated 90 degrees such that depth increases to the right.

        :param args: All arguments are passed to vfence. The results are then rotated to the horizontal orientation.
        :return:
        """
        out = self.vfence(*args)
        out.rot90()
        return out

    def vpole(self, *args):
        """
        Construct a "vertical pole" type template for visualising and corellating between one or more drillcores. This
        is identical to the hpole(...) layout, but rotated 90 degrees such that cores are vertical and depth increases
        downwards.

        :param args: All arguments are passed to hpole. The results are then rotated to vertical orientation.
        :return:
        """
        out = self.hpole(*args)
        # out.index = np.transpose(out.index, (1, 0, 2)) # rotate to vertical
        out.rot90()
        return out

    def _preprocessTemplates(self, depth_offsets, from_depth, groups, to_depth):
        # parse from_depth and to_depth if needed
        if from_depth is None:
            from_depth = np.min([np.min([t.from_depth for t in v]) for (k, v) in self.store.items()])
        if to_depth is None:
            to_depth = np.max([np.max([t.to_depth for t in v]) for (k, v) in self.store.items()])
        # ensure depth template keys are lower case!
        offs = {}
        for k, v in depth_offsets.items():
            offs[k.lower()] = v
        # crop templates to the relevant view area, and discard ones that do not fit
        cropped = {}
        for k, v in self.store.items():
            for T in v:
                assert T.from_depth is not None, "Error - depth info must be defined for template to be added."
                assert T.to_depth is not None, "Error - depth info must be defined for template to be added."
                T = T.crop(from_depth, to_depth, T.depth_axis, offs.get(k, 0))
                if T is not None:
                    # store
                    cropped[k.lower()] = cropped.get(k.lower(), [])
                    cropped[k.lower()].append(T)

        assert len(cropped) > 0, "Error - no templates are within depth range!"

        # sort templates by order in each group
        S = {}
        for k, v in cropped.items():
            S[k.lower()] = sorted(v)
        # resolve all unique paths
        paths = set()
        for k, v in S.items():
            for t in v:
                for b in t.boxes:
                    paths.add(os.path.join(t.root, b))
                    assert os.path.exists(
                        os.path.join(t.root, b)), "Error - one or more template directories do not exist?"
        paths = list(paths)
        # get group names to plot if not specified
        if groups is None:
            groups = list(S.keys())
        return S, from_depth, to_depth, groups, paths


    def add(self, group, template):
        """
        Add the specified template to this Canvas collection.

        :param group: The name of the group to add this template to.
        :param template: The template object.
        """
        self.__setitem__(group, template)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        """
        A shorthand way to add items to canvas.
        """
        assert isinstance(value, Template), "Error - only Templates can be added to a Canvas (for now...)"
        v = self.store.get(self._keytransform(key), [])
        v.append(value)
        self.store[self._keytransform(key)] = v

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key.lower()




