from src.config import *


def preprocess_image(img):
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - MEAN[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / STD[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img).cuda()
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def mkdirs():
    try:
        if os.path.exists('{}_plots'.format(opt.virus_type)):
            shutil.rmtree('{}_plots'.format(opt.virus_type))
        os.mkdir('{}_plots'.format(opt.virus_type))
        os.mkdir(TRAIN_PATH)
        os.mkdir(TEST_PATH)
        os.mkdir(TEST_XTICK_PATH)
        for c in CLASSES:
            os.mkdir(os.path.join('{}_plots'.format(opt.virus_type),
                                  c))
            os.mkdir(os.path.join(TRAIN_PATH,
                                  c))
            os.mkdir(os.path.join(TEST_PATH,
                                  c))
            os.mkdir(os.path.join(TEST_XTICK_PATH,
                                  c))
    except Exception as e:
        print(e)


def plots(path):
    df = pd.read_csv(path, index_col=False)
    new_df = df.drop(['pos1', 'pos2'], axis=1)
    label = new_df['label']
    new_f = new_df.drop(['label'], axis=1)
    columns = list(new_f.columns)
    x = new_f.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_f = pd.DataFrame(x_scaled)
    new_f.columns = columns
    new_f['label'] = label
    print(new_f)

    min_y = min(new_f.min(axis=1))
    max_y = max(new_f.max(axis=1))

    for idx, row in new_f.iterrows():
        label = int(row['label'])
        row = row.drop(['label'])
        xy = dict(zip(row.index, row.values))
        xy = {int(wvlength): spectrum for wvlength, spectrum in xy.items()}
        wvlength = list(xy.keys())
        spectrum = list(xy.values())
        fig = plt.figure(figsize=(20, 5))
        plt.plot(wvlength, spectrum, linewidth=3, c='black')
        # plt.ylim([min_y, max_y])
        fig.savefig(os.path.join(
            os.path.join('{}_plots'.format(opt.virus_type), CLASSES[label]),
            'xtick_sample{}_{}.png'.format(idx, label)),
            transparent=True)
        plt.axis('off')
        #plt.show()
        fig.savefig(os.path.join(
            os.path.join('{}_plots'.format(opt.virus_type), CLASSES[label]),
            'sample{}_{}.png'.format(idx, label)),
            transparent=True)


def split(plots_path):
    for c in CLASSES:
        c_path = os.path.join(plots_path, c)
        for img in os.listdir(c_path):
            if not img.startswith('xtick'):
                option = random.uniform(0, 1)
                src_path = os.path.join(c_path, img)
                train_path = os.path.join(
                    os.path.join(
                        os.path.join(plots_path,
                                     'trainset'),
                        c),
                    img)
                test_path = os.path.join(
                    os.path.join(
                        os.path.join(plots_path,
                                     'testset'),
                        c),
                    img)
                # 80/20 split
                if option < 0.8:
                    shutil.copyfile(src_path, train_path)
                elif 0.8 < option < 1:
                    shutil.copyfile(src_path, test_path)
                    shutil.copyfile(os.path.join(c_path, 'xtick_' + img),
                                    os.path.join(
                                        os.path.join(
                                            os.path.join(plots_path,
                                                         'testset_xtick'),
                                            c),
                                        img)
                                    )
        shutil.rmtree(c_path)


def update_dict(d1, d2):
    for k, v in d1.items():
        if k not in d2.keys():
            d2[k] = v
        else:
            d2[k] += v
    return d2


def avg_dict(d, divisor):
    d = {k: v / divisor for k, v in d.items()}
    return d


def visualize_1d_heatmap(masks, samples, feature_importances, feature_list, round, fold):
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    subpath = os.path.join(RESULT_PATH, opt.virus_type)
    if not os.path.exists(subpath):
        os.mkdir(subpath)

    plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])

    ax0.plot(feature_list, feature_importances, color='b')
    ax0.set_ylabel('Feature Importance')
    plt.xticks(range(int(min(feature_list)), int(max(feature_list)), 150))
    plt.xlim(min(feature_list), max(feature_list))

    ax0_1 = ax0.twinx()
    for c in masks.keys():
        ax0_1.plot(feature_list, samples[c], label=CLASSES[c])
    ax0_1.set_ylabel('Normalized Signal Intensity')
    ax0_1.tick_params(axis='y')

    plt.legend()
    plt.title('Round {}, Fold {}: {}'.format(round, fold, opt.virus_type))

    ax1 = plt.subplot(gs[1])
    all_mask = np.zeros(list(masks.values())[0].shape)
    for k in masks.keys():
        all_mask += masks[k]
    # all_mask /= np.mean([v.shape[0] for _, v in samples.items()])
    all_mask[all_mask < np.quantile(all_mask.flatten(), 0.7)] = 0
    all_mask = all_mask.transpose(1, 0)
    seaborn.heatmap(pd.DataFrame(np.uint8(255 * all_mask)), cbar=False, cmap='Reds')
    ax1.axis('off')
    print('{}/round_{}_fold_{}_{}.png'.format(subpath, round, fold, opt.virus_type))
    plt.savefig('{}/round_{}_fold_{}_{}.png'.format(subpath, round, fold, opt.virus_type))
    #plt.show()
