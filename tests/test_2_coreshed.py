import unittest

from hycore import get_sandbox, empty_sandbox, loadShed
import numpy as np
import hylite

import os

clean = False
class TestShed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Construct a directory containing dummy data for processing
        :return: a file path to the directory
        """
        # get sandbox directory
        if clean:
            empty_sandbox()
        cls.sandbox = get_sandbox()

        # try loading pre-existing shed
        if os.path.exists( os.path.join( cls.sandbox, 'eldorado.hdr' ) ):
            cls.S = loadShed(os.path.join( cls.sandbox, 'eldorado.hdr' ) )
        else:
            cls.S = get_sandbox(fill=True, vis=True, mosaic=True) # didn't work; build it

    @classmethod
    def tearDownClass(cls):
        # delete sandbox directory
        if clean:
            empty_sandbox()

    def test001_test_getters(self):

        # test getters before and after saving
        for i in range(2):
            holes = self.S.getHoles()
            self.assertListEqual( sorted([h.name for h in self.S.getHoles()]), ['H01', 'H02', 'H03'] )
            boxes = self.S.getBoxes()
            self.assertListEqual( sorted([b.name for b in self.S.getBoxes()]), ['b001', 'b001', 'b001', 'b002', 'b002', 'b003', 'b003', 'b004'] )
            for a in ['FENIX', 'LWIR']:
                self.assertTrue( a in boxes[0].getAttributes(ram_only=False) ) # check correct attributes are defined

            self.S.save()
            self.S.free()

    def test002_test_saved(self):
        # check that the correct file structure has been created
        self.assertTrue(os.path.exists(self.S.getDirectory()))
        self.assertTrue(os.path.exists(self.S.H02.getDirectory()))
        self.assertTrue(os.path.exists(self.S.H02.b001.getDirectory()))
        #self.assertTrue(os.path.exists(self.S.holes.H02.samples.s1.getDirectory()))

    def test003_test_load(self):
        from hycore import loadShed
        shed1 = loadShed(self.S.getDirectory()) # load with .shed path
        shed2 = loadShed( os.path.splitext(self.S.getDirectory())[0] + ".hdr" ) # load with .hdr path
        for shed in [shed1, shed2]:
            # print(len(shed.getBoxes()))
            self.assertGreater( len(shed.getBoxes()), 0 ) # make sure we retrieved some boxes...
            self.assertEqual( len(shed.getBoxes()), len(self.S.getBoxes() ) ) # check the sheds are the same

        print(len( shed.getBoxes() ) )

    def test004_test_vis(self):
        for b in self.S.getBoxes():
            b.updateVis(qaqc=False, thumb='FENIX')

    def test005_test_get(self):
        """
        Test overridden get function in hycore collections.
        """

        # get HSI data from this box
        b = self.S.getBoxes()[0]
        fenix = b.get('FENIX')
        self.assertGreater( fenix.band_count(), 10 ) # SWIR file should have many bands

        # get PNG data from this box
        fenixPNG = b.get('FENIX.png')
        self.assertEqual( fenixPNG.band_count(), 3 ) # PNG file should have 3 bands

        # get results data directly from box
        clay = b.get('BR_Clays')
        self.assertEqual( clay.band_count(), 3 )

    def test006_templates(self):
        """
        Test low level templating functions
        """
        from hycore.templates import unwrap_tray, unwrap_bounds, Template, Canvas, compositeStack, buildStack
        T = [] # templates from each hole
        C = Canvas()
        for i,h in enumerate(self.S.getHoles()):
            B = [] # blocks from each hole
            for j, b in enumerate( h.getBoxes() ):

                # get mask as binary image
                mask = hylite.HyImage(np.isfinite(b.mask.data[..., 0]))
                if (i == 0) and (j==0):
                    # run various combinations of template axis and mode
                    for method in ['blocks', 'sticks']:
                        for axis in [1,0]:
                            ix = unwrap_tray(mask, method=method, axis=axis)
                            self.assertGreater(len(np.unique(ix.data[..., 0])),
                                               100)  # more than 100 unique x-values in output
                            self.assertGreater(len(np.unique(ix.data[..., 1])),
                                               100)  # more than 100 unique y-values in output

                # create template and store
                template = Template(b.getDirectory(), unwrap_bounds(mask, pad=1), b.start, b.end) # just check this runs...
                template = Template(b.getDirectory(), unwrap_tray( mask, method='sticks', axis=0 ), b.start, b.end )

                # print(b.start, b.end)
                B.append( template ) # store in array for each hole
                C.add( b.holeName(), template ) # also add to our canvas for more complex layout

                if (i == 0) and (j==0):
                    # check to / from image
                    img = B[-1].toImage()
                    self.assertEqual( img.band_count(), 3 )
                    _B = Template.fromImage( img )
                    self.assertListEqual( _B.boxes, B[-1].boxes )
                    self.assertEqual( _B.root, B[-1].root )
                    self.assertTrue( ( _B.index == B[-1].index).all() )

            # sort blocks by depth and stack in template
            Ts = Template.stack(sorted(B), axis=0)
            T.append(Ts)

            # unwrap each core and test stack function
            preview = compositeStack(Ts, ['FENIX', 'LWIR', 'BR_Clays'], [hylite.SWIR, hylite.LWIR, (0, 1, 2)], vmin=2, vmax=98)
            preview.data = (preview.data * 255).astype(np.uint8)
            h.results.preview = preview
            h.results.save()

            #print([b.center_depth for b in sorted( B )] )
            self.assertEqual(Ts.root, h.getDirectory()) # check root directory is correct

        # test stacking directly from boxes
        Ttest = buildStack( h.getBoxes(), pad=2, axis=0, transpose=True )
        Itest = Ttest.apply('FENIX.png')
        Ttest.add_outlines( Itest )
        fig,ax = Ttest.quick_plot((0,1,2))
        Ttest.add_ticks(ax)

        self.assertTrue(Itest.xdim() > 100)
        self.assertTrue( np.isfinite( Itest.data ).any() )

        # test stacking with templates
        TStacked = Template.stack( T, axis=1 )
        self.assertEqual(TStacked.root, self.S.getDirectory() ) # check root directory is correct
        self.assertGreater( len(np.unique(TStacked.index[..., 1] ) ), 100 ) # more than 100 unique x-values in output
        self.assertGreater(len(np.unique(TStacked.index[..., 2])), 100)  # more than 100 unique y-values in output

        # test stacking with subsample
        TStacked2 = Template.stack(T, axis=1, xstep=2, ystep=2 )
        self.assertEqual( int(TStacked.index.shape[0] / TStacked2.index.shape[0] ) , 2 ) # check result is half the size
        self.assertEqual( int(TStacked.index.shape[0] / TStacked2.index.shape[0]), 2 ) # check result is half the size

        # test stacking with Canvas
        TStacked3 = C.hpole( from_depth = 2.5, to_depth = 17.5, res=2e-3, scaled=True, groups=['H01','H03','H02'] )

        # save index image for manual checks
        for T,n in zip( [TStacked, TStacked2, TStacked3], ['index1', 'index2', 'index3']):
        #for T, n in zip([TStacked3], ['index3']):
            ix = hylite.HyImage(T.index[..., ].astype(np.float32))
            ix.percent_clip(2, 98, per_band=True)
            ix.data = np.clip(ix.data*255, 0, 255).astype(np.uint8)
            self.S.results.set( n, ix )

            # generate mosaic image
            swir = T.apply('FENIX', bands=hylite.SWIR, xstep=3, ystep=2 )
            r = swir.percent_clip(2,98)
            #self.assertGreater( r[1], 0.9 ) # check there are some real data!
            swir.data = np.clip(swir.data*255, 0, 255).astype(np.uint8) # convert to uint 8
            self.S.results.set('%s_swir_mosaic'%n, swir )

            # generate mosaic image from results
            clays = T.apply('BR_Clays', bands=(0,1,2), strict=False ) # N.B. clay BR is missing from one box! ;-)
            #self.assertGreater(np.nanpercentile(clays.data, 99), 250 ) # check there are some real data!
            self.S.results.set("%s_clay_mosaic" % n, clays )

            # check mosaic subsampling worked
            self.assertLess( np.abs( swir.xdim() * 3- clays.xdim() ), 3 )
            self.assertLess( np.abs( swir.ydim() * 2 - clays.ydim()), 3 )

            # run gridding code
            T.getGrid()



        self.S.results.save() # save results as PNG for manual checking

    def test007_templates2(self):
        """
        Test templating functions implemented in the Shed, Core and Box classes.
        """
        for i, h in enumerate(self.S.getHoles()):
            # check box templates
            for j, b in enumerate(h.getBoxes()):
                for a in [0,1]:
                    template = b.getTemplate(method='sticks', axis=a )
                    self.assertTrue( template.depths is not None )
                    self.assertTrue( template.depth_ticks is not None )
                    template.quick_plot(1, rot=False, path=os.path.join(b.results.getDirectory(), 'T%d.png'%a)) # save image for checking

            # get hole template
            for  m in ['pole', 'fence']:
                T = h.getTemplate(method=m, res=2e-3)
                T.quick_plot(1, rot=False, interval=0.5,
                                 path=os.path.join(h.results.getDirectory(), '%s.png'%m) )  # save image for checking

        # get shed template
        for m in ['pole', 'fence']:
            T = self.S.getTemplate(method=m, res=2e-3)
            T.quick_plot(1, rot=False, interval=0.5,
                         path=os.path.join(self.S.getDirectory(), '%s.png' % m))  # save image for checking

    def test008_automosaic(self):
        # just run; output needs to be manually checked but should be fine
        self.S.updateMosaics(res=2e-3, files=['FENIX.png', 'LWIR.png', 'BR_Clays.png'], outline = 0.2 )

        # check dims
        for h in self.S.getHoles():
            for m in ['pole', 'fence']:
                T = h.results.get(m).template
                if 'pole' in m:
                    zd = T.xdim()
                else:
                    zd = T.ydim()
                self.assertEqual(zd, len(T.header.get_list('depths')) )

    def test009_testIDs(self):
        box = self.S.getBoxes()[0]
        self.assertEqual(box.getFullName(), 'eldorado/H01/b001')

    def test010_testAbout(self):
        self.S.createAboutMD(author_name='Sam Thiele')
        self.assertTrue( os.path.exists( os.path.join(self.S.getDirectory(), 'about.md')))

    def test011_quantize(self):
        boxes = self.S.getBoxes()
        for b in boxes:
            b.free()
            b.quantize(vthresh=5, subsample=50, smooth=0)
        for s in b.getSensors():
            self.assertTrue( os.path.exists( os.path.join( b.getDirectory(), 'spectra.hyc/%s_idx.hdr'%s) ) )

    def test012_unquantize(self):
        for b in self.S.getBoxes():
            b.unquantize(sensors='FENIX', save=True)
            b.updateVis( sensors='FENIX', qaqc=False )

    def test013_exportQuanta(self):
        pth = os.path.join( os.path.dirname( self.S.getDirectory() ), 'exported' )
        self.S.exportQuanta(path=pth, clean=True, crop=True, results=['BR_Clays'], ss=2)
        self.assertTrue( os.path.exists( os.path.join( pth, 'H01' ) ) )
        
if __name__ == '__main__':
    unittest.main()
