from src.config import *


def get_name(filename):
    return filename.split('.')[0]


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if d.endswith(EXT)]
    classes = list(map(get_name, classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def merge_df(dataframe, label, cnt_wave):
    dataframe.columns = ['pos1', 'pos2', 'wavenumber', 'signal']

    CNT_ROW = len(dataframe.index) // cnt_wave

    df = dataframe[0:cnt_wave]

    df_mergerd = pd.DataFrame({'pos1': [df.iloc[0]['pos1']], 'pos2': [df.iloc[0]['pos2']]})
    for idx_wave in range(cnt_wave):
        wavenumber = df.iloc[idx_wave]['wavenumber']
        signal = df.iloc[idx_wave]['signal']

        df_mergerd[str(int(wavenumber))] = signal
        df_mergerd['label'] = label

    cnt = 0
    for idx in range(1, CNT_ROW):

        df = dataframe[idx * cnt_wave:(idx + 1) * cnt_wave]

        df_new = pd.DataFrame({'pos1': [df.iloc[0]['pos1']], 'pos2': [df.iloc[0]['pos2']]})
        cnt_dup = 1
        for idx_wave in range(cnt_wave):
            wavenumber = df.iloc[idx_wave]['wavenumber']
            signal = df.iloc[idx_wave]['signal']

            if str(int(wavenumber + 1)) in df_new.columns and signal > 1.5 * df_new[str(int(wavenumber) + 1)].values[0]:

                signal = df_new[str(int(wavenumber) + 1)].values[0]
            elif str(int(wavenumber - 1)) in df_new.columns and signal > 1.5 * df_new[str(int(wavenumber) - 1)].values[
                0]:
                signal = df_new[str(int(wavenumber) - 1)].values[0]


            if str(int(wavenumber)) in df_new.columns:
                cnt_dup += 1
                df_new[str(int(wavenumber))] = (df_new[str(int(wavenumber))] * (cnt_dup - 1) + signal) / cnt_dup
            else:

                cnt_dup = 1
                df_new[str(int(wavenumber))] = signal
            df_new['label'] = label
        df_mergerd = pd.concat([df_mergerd, df_new], sort=False)
        if cnt % 10 == 0: print('{:.2f}% has done for {}'.format(cnt / CNT_ROW * 100, CLASSES[label]))
        cnt += 1
    print(df_mergerd.shape)
    return df_mergerd


def run():
    classes, class_to_idx = find_classes(ROOT)
    print(classes)
    print(class_to_idx)
    file_final = pd.DataFrame()

    for filename in [c+'.txt' for c in CLASSES]:

        file = pd.read_csv(ROOT + filename, sep="\t", header=None)
        cnt = 0
        pos1 = file.iloc[0][0]
        pos2 = file.iloc[0][1]
        for idx in range(int(1e5)):
            row = file.iloc[idx]
            if row[0] == pos1 and row[1] == pos2:
                cnt += 1
            else:
                break
        label = class_to_idx[filename.split('.txt')[0]]

        file_final = pd.concat([file_final, merge_df(file, label, cnt)], sort=False)
        print(file_final)

    file_final1 = file_final.replace('', np.nan)
    file_final1 = file_final1.dropna(axis=1)
    file_final.to_csv(ROOT + 'feature.csv', sep=",", index=False)

    print(file_final1.shape)