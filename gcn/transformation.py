import numpy as np


class RemoveParAndFlopTransform:
    def __call__(self, graph):
        if graph.x is not None:
            # Remove the columns of features include parameters and FLOPs
            graph.x = np.delete(graph.x, [7, 8], 1)

        return graph


class OneHotToIndexTransform:
    def __call__(self, graph):
        if graph.x is not None:
            # Remove the columns of features include parameters and FLOPs
            try:
                tmp = np.delete(graph.x, [7, 8], 1)
            except:
                tmp = graph.x

            # set position 0 as the feature index
            graph.x[:, 0] = np.argmax(tmp, axis=1)
            # remove one-hot (position 1 to 6)
            graph.x = np.delete(graph.x, range(1, 7), 1)

        return graph