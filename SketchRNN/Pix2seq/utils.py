import os
import urllib.request
import numpy as np
import cairosvg
from six.moves import range
import svgwrite
from PIL import Image


def download_dataset(class_name: str):
    dirname = 'npz'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    url = f'https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{class_name}.npz'
    urllib.request.urlretrieve(url, os.path.join(dirname, f'{class_name}.npz'))


def merge_npz_files(files, axis=0):
    """
    Merge multiple .npz files into a single .npz file.

    Args:
        files: List of .npz files to merge.
        output_file: Name of the output .npz file.
        axis: Axis along which to concatenate the arrays with the same key.

    Returns:
        None
    """

    # Initialize an empty dictionary to store the arrays
    arrays = {}

    # Loop through the list of files
    for file in files:
        # Load the arrays from the .npz file
        loaded_arrays = np.load(file, encoding='latin1', allow_pickle=True)

        # Loop through the arrays in the .npz file
        for key in loaded_arrays:
            # If the key is not in the dictionary, add it
            if key not in arrays:
                arrays[key] = loaded_arrays[key]
            # If the key is already in the dictionary, append the new array to the existing one
            else:
                arrays[key] = np.concatenate((arrays[key], loaded_arrays[key]), axis=axis)

    return arrays

def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return min_x, max_x, min_y, max_y


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:  # stroke: [N_points, 3]
        ml = len(stroke)
        max_len = ml if ml > max_len else max_len

    return max_len


def draw_strokes(data, svg_filename, factor=0.2, padding=50):
    """
    little function that displays vector images and saves them to .svg
    :param data:
    :param factor:
    :param svg_filename:
    :param padding:
    :return:
    """
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (padding + max_x - min_x, padding + max_y - min_y)
    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = int(padding / 2) - min_x
    abs_y = int(padding / 2) - min_y
    p = "M%s, %s " % (abs_x, abs_y)
    # use lowcase for relative position
    command = "m"

    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + ", " + str(y) + " "
    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    #dwg.save()

    svg_code = dwg.tostring()
    img = cairosvg.svg2png(bytestring=svg_code)
    image = Image.open(io.BytesIO(img))
    image = image.resize((48,48))
    #image = image.convert('1')
    aarr = np.asarray(image)
    aarr = np.where(aarr < 255, 0, 255)
    aarr = aarr[:,:,1]
    # aarr = np.reshape(aarr, (28*28))
    # np.save('array', aarr)
    #image.save(svg_filename + '.png')
    return dims, dwg.tostring(), aarr


def load_dataset(data_dir, data_set, percentage):
    """Loads the .npz file, and splits the set into train/valid/test."""

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.
    img_H, img_W = 48, 48

    if isinstance(data_set, list):
        datasets = data_set
    else:
        datasets = [data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    png_paths_map = {'train': [], 'valid': [], 'test': []}

    for dataset in datasets:
        data_filepath = os.path.join(data_dir, 'npz', dataset)
        data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        print('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']),
            dataset))
        if train_strokes is None:
            train_strokes = data['train'][:int(len(data['train']) * percentage)]  # [N (#sketches),], each with [S (#points), 3]
            valid_strokes = data['valid'][:int(len(data['valid']) * percentage)] 
            test_strokes = data['test'][:int(len(data['test']) * percentage)] 
        else:
            train_strokes = np.concatenate((train_strokes, data['train'][:int(len(data['train']) * percentage)] ))
            valid_strokes = np.concatenate((valid_strokes, data['valid'][:int(len(data['valid']) * percentage)] ))
            test_strokes = np.concatenate((test_strokes, data['test'][:int(len(data['test']) * percentage)] ))

        splits = ['train', 'valid', 'test']
        for split in splits:
            for im_idx in range(len(data[split][:int(len(data[split]) * percentage)] ) ):
                png_path = os.path.join(data_dir, 'png', dataset[:-4], split,
                                        str(img_H) + 'x' + str(img_W), str(im_idx) + '.png')
                png_paths_map[split].append(png_path)

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len)))
    assert len(train_strokes) == len(png_paths_map['train'])
    assert len(valid_strokes) == len(png_paths_map['valid'])
    assert len(test_strokes) == len(png_paths_map['test'])

    # calculate the max strokes we need.
    max_seq_len = get_max_len(all_strokes)

    result = {'train':train_strokes, 'valid': valid_strokes, 'test': test_strokes}
    png_paths = {'train':png_paths_map['train'], 'valid':png_paths_map['valid'], 'test':png_paths_map['test']}
    return result, png_paths


def pad_image(png_filename, pngsize):
    curr_png = Image.open(png_filename).convert('RGB')
    png_curr_w = curr_png.width
    png_curr_h = curr_png.height
    print("pngsize: {}, {}".format(pngsize[0],pngsize[1]))

    if png_curr_w != pngsize[0] and png_curr_h != pngsize[1]:
        print('Not aligned', 'png_curr_w', png_curr_w, 'png_curr_h', png_curr_h)

    padded_png = np.zeros(shape=[pngsize[1], pngsize[0], 3], dtype=np.uint8)
    padded_png.fill(255)

    if png_curr_w > png_curr_h:
        pad = int(round((png_curr_w - png_curr_h) / 2))
        padded_png[pad: pad + png_curr_h, :png_curr_w, :] = np.array(curr_png, dtype=np.uint8)
    else:
        pad = int(round((png_curr_h - png_curr_w) / 2))
        padded_png[:png_curr_h, pad: pad + png_curr_w, :] = np.array(curr_png, dtype=np.uint8)

    padded_png = Image.fromarray(padded_png, 'RGB')
    padded_png.save(png_filename, 'PNG')


def svg2png(dwg_string, svgsize, pngsize, png_filename, padding=False):
    """convert svg into png, using cairosvg"""
    svg_w, svg_h = svgsize
    png_w, png_h = pngsize
    x_scale = png_w / svg_w
    y_scale = png_h / svg_h

    if x_scale > y_scale:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_height=png_h)
    else:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_width=png_w)

    if padding:
        pad_image(png_filename, pngsize)
    img = Image.open(png_filename)
    img = img.resize((48,48))
    img = np.asarray(img)
    img = np.where(img < 254, 0, 255).astype(np.uint8)
    #plt.imshow(img)
    img = Image.fromarray(img)
    img.save(png_filename, 'PNG')

def render_svg2bitmap(data_base_dir, data_set, percentage):
    img_H, img_W = 250, 250

    npz_dir = os.path.join(data_base_dir, 'npz')
    svg_dir = os.path.join(data_base_dir, 'svg')
    png_dir = os.path.join(data_base_dir, 'png')

    #model_params = sketch_rnn_model.get_default_hparams()
    
    for dataset_i in range(len(data_set)):
        assert data_set[dataset_i][-4:] == '.npz'
        cate_svg_dir = os.path.join(svg_dir, data_set[dataset_i][:-4])
        cate_png_dir = os.path.join(png_dir, data_set[dataset_i][:-4])

        datasets, png_paths = load_dataset(data_base_dir, data_set, percentage)

        data_types = ['train', 'valid', 'test']
        for data_type in data_types:
            split_cate_svg_dir = os.path.join(cate_svg_dir, data_type)
            split_cate_png_dir = os.path.join(cate_png_dir, data_type,
                                              str(48) + 'x' + str(48))

            os.makedirs(split_cate_svg_dir, exist_ok=True)
            os.makedirs(split_cate_png_dir, exist_ok=True)

            split_dataset = datasets[data_type]

            for ex_idx in range(len(split_dataset)):
                stroke = np.copy(split_dataset[ex_idx])
                print('example_idx', ex_idx, 'stroke.shape', stroke.shape)
                print('len', len(split_dataset))

                png_path = png_paths[data_type][ex_idx]
                # print("HOHOHO")
                # print(split_cate_png_dir)
                # print(png_path[:len(split_cate_png_dir)])
                # assert split_cate_png_dir == png_path[:len(split_cate_png_dir)]
                actual_idx = png_path[len(split_cate_png_dir) + 1:-4]
                svg_path = os.path.join(split_cate_svg_dir, str(actual_idx) + '.svg')
                
                print(png_path)
                svg_size, dwg_bytestring, aarr = draw_strokes(stroke, svg_path, padding=10)  # (w, h)
                svg2png(dwg_bytestring, svg_size, (img_W, img_H), png_path,
                               padding=True)