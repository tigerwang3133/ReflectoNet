from src.utils.util import *


class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.net, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)

        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.net.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # gradient of the last RELU
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 1x64xwxh or 1x64x1
        # output after the last RELU
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]  # (64,1)
        # weights = np.mean(grads_val, axis=(2, 3))[0, :]
        weights = np.mean(grads_val, axis=2)[0, :]  # (64,)
        # cam = np.zeros(target.shape[1:], dtype=np.float32)
        cam = np.zeros(target.shape, dtype=np.float32)
        for i, w in enumerate(weights):
            # cam += w * target[i, :, :]
            cam[i, :] = w * target[i, :]

        cam = np.maximum(cam, 0)
        # size of the spectra line plot 2000x500
        # cam = cv2.resize(cam, (2000, 500))
        cam = cv2.resize(cam, (50, 1980))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


def grab_cam(opt):
    save_path = opt.virus_type + '_grad_cam_examples'
    grad_cam = GradCam(model=torch.load(os.path.join('saved_pkls', 'nn.pkl')), \
                       target_layer_names=['12'], use_cuda=True)  # 13 -> the last RELU

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    for c in CLASSES:
        os.mkdir(os.path.join(save_path, c))

    visualize_path = '{}_plots/testset'.format(opt.virus_type)
    for c in os.listdir(visualize_path):
        c_path = os.path.join(visualize_path, c)
        for img_name in os.listdir(c_path):
            f_path = os.path.join(c_path, img_name)
            img = cv2.imread(f_path, 1)
            xtick_path = '/'.join(f_path.split('/')[:-3]) + '/testset_xtick/' + '/'.join(f_path.split('/')[-2:])
            ori_img = cv2.imread(xtick_path, 1)
            img = np.float32(img) / 255
            ori_img = np.float32(ori_img) / 255
            input = preprocess_image(img)
            target_index = None
            mask = grad_cam(input, target_index)
            cam_on_image = show_cam_on_image(ori_img, mask)
            cv2.imwrite(os.path.join(os.path.join(save_path,
                                                  c),
                                     img_name), cam_on_image)
