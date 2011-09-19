#!/usr/bin/env python

import logging

import numpy as np

import pywaveclus

def test_clusters_to_indices():
    ids = pywaveclus.cluster.spc.clusters_to_indices([[1,2,3],[4,5,0,6]])
    assert all(ids == np.array([1, 0, 0, 0, 1, 1, 1])), "clusters_to_indicies failed"
    
    try:
        pywaveclus.cluster.spc.clusters_to_indices([[1,2,3],[4,5,6]]) # missing index 0
        raise Exception("clusters_to_indices failed to raise IndexError on missing index")
    except IndexError:
        pass # this is what it's supposed to do
    except Exception as e:
        raise e

def test_spc(plot=False):
    fakefeatures = np.array([\
    [9.121892737849592869e-01,1.702694147769417965e-01,1.976813384432012377e+00],
    [1.505255486133956655e+00,-2.132262451773800915e-01,1.139754580088775171e+00],
    [1.796021056608481992e+00,5.482791889945581865e-02,1.199430447971401303e+00],
    [1.766677651808613581e+00,-2.355300788060969985e-01,1.486193183085017466e+00],
    [1.600278741356271794e+00,-1.775991502788796250e-01,1.527281967784109185e+00],
    [9.670918210696861639e-01,2.486423132958039961e-01,1.961735764050466013e+00],
    [1.734223899292208548e+00,-2.512813368026616256e-01,1.296319560299583173e+00],
    [1.094575833590590630e+00,1.267858197785596275e-01,1.779466395716537086e+00],
    [1.059276514121765711e+00,1.305797159406407726e-01,2.009464406413342452e+00],
    [1.674525742681307161e+00,-2.156485144335169579e-01,1.303970246832113933e+00],
    [9.501799423598481509e-01,2.315823895577806546e-01,1.768562216091043693e+00],
    [1.041072987670736083e+00,-4.213460119602729925e-02,2.121463958816212081e+00],
    [1.836259659903269004e+00,-2.598869543998592047e-01,1.459478778381221531e+00],
    [1.714749555013693305e+00,-7.457626088312108159e-02,1.289433351235938607e+00],
    [9.027047382404921327e-01,5.136644878710594497e-02,1.936821952740319208e+00],
    [1.074105537844705749e+00,6.360771785660634947e-02,1.838655582554846735e+00],
    [1.643212005392947983e+00,-2.745992386999307477e-01,1.297965407475970112e+00],
    [9.489769736806544786e-01,2.426835164045688664e-01,1.894859490065952112e+00],
    [1.734098755661226399e+00,-3.926875607205684293e-01,1.317626529276785918e+00],
    [9.882055268780466362e-01,1.558599301169458773e-01,1.934077210710049455e+00],
    [1.808304977297616301e+00,-2.198704001991904633e-01,1.501091799349095757e+00],
    [1.764705842536313352e+00,-3.250740305534627339e-01,1.516096375539988550e+00],
    [1.646630712538638353e+00,-2.826626520191597436e-01,1.249581548493040595e+00],
    [1.683388848533310611e+00,-2.002866528469764429e-01,1.346262890851235516e+00],
    [1.756721094225609958e+00,-2.730115827316190735e-01,1.112886885501471523e+00],
    [1.669121401396321591e+00,-2.186152753041630259e-01,1.435177756094649260e+00],
    [1.056231117027902400e+00,1.871922006208025513e-01,1.890754987792580444e+00],
    [1.629917946089443292e+00,-1.795630738890776623e-01,1.218610980156035417e+00],
    [1.603581822150865976e+00,-2.819810598942671387e-01,1.460395310442928807e+00],
    [1.661431343339078603e+00,-1.971581615096433815e-01,1.393247354066568855e+00]])
    
    
    logging.basicConfig(level=logging.DEBUG)
    
    features = fakefeatures
    # features = realfeatures
    
    clusters, (cdata, tree) = pywaveclus.cluster.spc.cluster(features)#, minclus = 15)
    logging.info("Clusters: %s" % str([len(c) for c in clusters]))
    # print len(unmatched)
    
    if plot:
        import pylab as pl
        from mpl_toolkits.mplot3d import Axes3D
        
        ax = Axes3D(pl.figure())
        colors = ['k','b','g','r','m','y']
        for (i,c) in enumerate(clusters):
            d = features[c]
            if len(d) == 0: continue
            ax.scatter(d[:,0],d[:,1],d[:,2],c=colors[i])
        
        # d = features[unmatched]
        # ax.scatter(d[:,0],d[:,1],d[:,2],c='k')
        
        pl.figure()
        
        nf = features.shape[1]
        for x in xrange(nf):
            for y in xrange(nf):
                if x > y:
                    pl.subplot(nf,nf,x+y*nf+1)
                    for (i, c) in enumerate(clusters):
                        d = features[c]
                        if len(d) == 0: continue
                        pl.scatter(d[:,x], d[:,y], c=colors[i], s=10, alpha=0.5, edgecolors=None)
                        # pl.scatter(features[:,x], features[:,y], s=1)
                    # pl.gca().set_axis_off()
                    pl.gca().set_xticks([])
                    pl.gca().set_yticks([])
                    b = 1/8.
                    xr = features[:,x].max() - features[:,x].min()
                    pl.xlim(features[:,x].min()-xr*b, features[:,x].max()+xr*b)
                    yr = features[:,y].max() - features[:,y].min()
                    pl.ylim(features[:,y].min()-yr*b, features[:,y].max()+yr*b)
        
        pl.figure()
        pl.imshow(tree, interpolation='nearest')
        pl.figure()
        pl.imshow(cdata, interpolation='nearest')
        pl.show()
    return
