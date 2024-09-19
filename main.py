from MOT import mot
from Pointnet2 import train, prediction
from SemanticSceneRestriction import srr



def main():
    semantic_features = ['v', 'RCS']  #'v', 'RCS'
    withSRR = True
    WithSemantic = True
    plot_vicon=True
    plot_ghost=True

    srr(semantic_features=semantic_features,withSRR=withSRR)
    
    # train(semantic_features=semantic_features)
    prediction(semantic_features=semantic_features,withSRR=withSRR)

    mot(semantic_features=semantic_features, WithSemantic=WithSemantic, withSRR=withSRR, plot_vicon=plot_vicon,plot_ghost=plot_ghost)



if __name__ == '__main__':
    main()
