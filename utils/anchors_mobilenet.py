import numpy as np


class AnchorBoxes:

    def __init__(self,  img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1],**kwargs):

        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        
        self.flip = flip
        self.variances = np.array(variances)
    
    def call(self, x, mask=None):
        
        img_width = self.img_size[1]
        img_height = self.img_size[0]
        box_heights = []
        box_widths = []
        feature_map_height = x[0]
        feature_map_width = x[1]

        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 0:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        
        box_widths = np.array(box_widths) * 0.5
        box_heights = np.array(box_heights) * 0.5
        
        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        
        step_height = img_height / feature_map_height
        step_width = img_width / feature_map_width
        
        #line_x = np.linspace(offset_width * step_width, img_width - offset_width * step_width, feature_map_width)
        #line_y = np.linspace(offset_height * offset_height, img_height - offset_height * step_height, feature_map_height)
        linx = np.linspace(0.5 * step_width, img_width - 0.5 *step_width,
                           feature_map_height)
        liny = np.linspace(0.5 * step_height, img_height - 0.5 * step_height,
                           feature_map_height)
        center_x, center_y = np.meshgrid(linx, liny)
        center_x = center_x.reshape(-1, 1)
        center_y = center_y.reshape(-1, 1)

        # Every prior_boxes need two boxes, one is be used (xmin, ymin), the other is be used (xmax, ymax) 
        num_priors = len(self.aspect_ratios)
        prior_boxes = np.concatenate((center_x, center_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
        #print(num_priors)
        # Compute four corners
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        # Normalize
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)

        return prior_boxes
        
def get_anchors_300(image_size=(300, 300), anchors=[30, 60, 111, 162, 213, 264, 315]):

    if image_size != (300, 300):
        raise ValueError("This anchor need to used (300, 300).")

    features_map_length = [19, 10, 5, 3, 2, 1] 
    variances = [0.1, 0.1, 0.2, 0.2]
    
    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                             variances=variances)
    conv4_3_norm_mbox_priorbox = priors.call((features_map_length[0],features_map_length[0]))
    
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2, 3],
                        variances=variances)
    fc7_mbox_priorbox = priors.call((features_map_length[1],features_map_length[1]))
    
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[2], max_size=anchors[3],aspect_ratios=[2, 3],
                        variances=variances)
    conv6_2_mbox_priorbox = priors.call((features_map_length[2],features_map_length[2]))
    
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[3], max_size=anchors[4],aspect_ratios=[2, 3],
                        variances=variances)
    conv7_2_mbox_priorbox = priors.call((features_map_length[3],features_map_length[3]))
    
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[4], max_size=anchors[5],aspect_ratios=[2],
                        variances=variances)
    conv8_2_mbox_priorbox = priors.call((features_map_length[4],features_map_length[4]))
    
    priors = AnchorBoxes(img_size=image_size, min_size=anchors[5], max_size=anchors[6],aspect_ratios=[2],
                        variances=variances)
    conv9_2_mbox_priorbox = priors.call((features_map_length[5],features_map_length[5]))
    
    mbox_priorbox = np.concatenate([conv4_3_norm_mbox_priorbox,
                                    fc7_mbox_priorbox,
                                    conv6_2_mbox_priorbox,
                                    conv7_2_mbox_priorbox,
                                    conv8_2_mbox_priorbox,
                                    conv9_2_mbox_priorbox], axis=0)
    
    return mbox_priorbox