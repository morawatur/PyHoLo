# '<i' == little endian byte ordering
# stworzyc klase tag_group, tag itp.

# sizes of different datatypes in bytes
type_size = { 'char': 1, 'bool': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'float': 4, 'double': 8 }

# format strings for different datatypes to be used inside the struct.unpack() function
type_format = { 'char': 'c', 'uchar': 'B', 'bool': '?', 'i16': 'h', 'ui16': 'H', 'i32': 'i', 'ui32': 'I', 'float': 'f', 'double': 'd' }

# tag labels for values which we want to read (image dimensions, pixel dimensions,
# units of pixel dimensions and image data, i.e. pixel values)
tags_data = { 'RestoreImageDisplayBounds': [], 'Scale': [], 'Units': '', 'Data': [] }

#-----------------------------------------------------------------------------------

def Reverse(word):
  '''Returns reversed string.
  Keyword arguments:
  word (string) -- string to be reversed
  '''
  return word[::-1]

#-----------------------------------------------------------------------------------

def GetTypeSize(type_id):
  '''Gives size of the datatype in bytes.
  Keyword arguments:
  type_id (integer) -- id number of the datatype given by dm3 format
  Returns: integer
  '''

  if type_id == 0:
    return 0
  elif type_id == 2 or type_id == 4:
    return type_size['i16']
  elif type_id == 3 or type_id == 5:
    return type_size['i32']
  elif type_id == 6:
    return type_size['float']
  elif type_id == 7:
    return type_size['double']
  elif type_id == 8:
    return type_size['bool']
  elif type_id == 9:
    return type_size['char']
  elif type_id == 10:
    return type_size['i8']

#------------------------------------------------------------------------------------

def ReadDm3File(dm3_fpath):
  '''Reads dm3 file byte after byte to get the image data.
  Then saves the image in png format.
  Possible future developments: reading and storing values of all dm3 tags.
  Keyword arguments:
  dm3_fpath (string) -- path of the dm3 file to be read
  Returns: None
  '''

  import sys
  import struct
  import numpy as np

  sys.stdout = open(dm3_fpath.replace('.dm3', '_log.txt'), 'w')
  dm3_file = open(dm3_fpath, 'rb')
  print('Reading DM3 File...')

  header_size = 3 * type_size['i32']
  header = dm3_file.read(header_size)
  header = Reverse(header)
  # zamiast Reverse() mozna wczytac stringa w odpowiednim formacie, np.
  # header = struct.unpack('>i', dm3_file.read(size))

  header_format = '%di' % (len(header) // 4)
  header_list = list(struct.unpack(header_format, header))

  dm3_items = { 'dm_version': 0, 'file_size': 0, 'byte_order': 0, 'tag_group': 0 }

  dm3_items['dm_version'] = header_list[2]
  dm3_items['file_size'] = header_list[1]
  dm3_items['byte_order'] = header_list[0]

  print('DM version: ' + str(dm3_items['dm_version']) + '\n' \
        'File size: ' + str(dm3_items['file_size']) + ' bytes')

  main_tag_group_size = dm3_items['file_size'] - header_size
  ReadTagGroup(dm3_file)

  image_dims = tags_data['RestoreImageDisplayBounds']
  pixel_dims = tags_data['Scale']
  image_data = tags_data['Data']

  # SaveDm3AsPng(image_data, image_dims, dm3_fpath)
  print('\nAll done')
  sys.stdout = sys.__stdout__

  image1d = np.asarray(image_data)
  image2d = np.reshape(image1d, tuple(image_dims))
  return image2d, pixel_dims

#-----------------------------------------------------------------------------------

def ReadTagGroup(dm3_file):
  '''Reads group of dm3 tags.
  For every single tag in a group it calls ReadTag() function.
  Keyword arguments:
  dm3_file (file) -- dm3 file object
  image_data (list) -- empty container for the image data
  (image data will be stored in this list when tag with 'Data' label will be found)
  Returns: None
  '''

  import struct

  print('\n----------------------------------------\n' + \
        'Tag Group' + \
        '\n----------------------------------------')

  tgroup_header_size = 2 * type_size['bool'] + type_size['i32']
  tgroup_header = dm3_file.read(tgroup_header_size)
  tgroup_header = Reverse(tgroup_header)

  tgroup_items = { 'is_sorted': False, 'is_open': False, 'n_tags': 0, 'tag_list': [] }

  tgroup_items['n_tags'] = struct.unpack('i', tgroup_header[:4])[0]
  tgroup_items['is_open'] = struct.unpack('?', tgroup_header[4:5])[0]
  tgroup_items['is_sorted'] = struct.unpack('?', tgroup_header[5:6])[0]

  for tag_idx in range(0, tgroup_items['n_tags']):    # moze byc tez range(tgroup_items['n_tags'])
    ReadTag(dm3_file)

#-----------------------------------------------------------------------------------

def ReadTag(dm3_file):
  '''Reads single tag.
  If a tag turns out to be a tag group it calls ReadTagGroup() function.
  If a tag is a single tag then it calls ReadTagType() function.
  Keyword arguments:
  dm3_file (file) -- dm3 file object
  image_data (list) -- container for the image data
  Returns: None
  '''

  import struct

  tag_header_size = type_size['char'] + type_size['i16']
  tag_header = dm3_file.read(tag_header_size)
  tag_header = Reverse(tag_header)

  tag_items = { 'is_group': False, 'label_length': 0, 'label': '', 'tag_content': [] }

  tag_items['label_length'] = struct.unpack('h', tag_header[:2])[0]
  is_data_or_group = (struct.unpack('c', tag_header[2:3])[0]).decode('utf-8')

  tag_items['is_group'] = True if is_data_or_group == chr(20) else False

  label_format = '%ds' % tag_items['label_length']
  tag_items['label'] = (struct.unpack(label_format, dm3_file.read(tag_items['label_length']))[0]).decode('utf-8', 'ignore')

  print('"%s"' % (tag_items['label']))
  #print(str(tag_items['label'])[1:])

  if tag_items['is_group']:
    ReadTagGroup(dm3_file)
  else:
    ReadTagType(dm3_file, tag_items['label'])

#-----------------------------------------------------------------------------------

def ReadTagType(dm3_file, tag_label):
  '''Reads information about data structure and datatypes of individual values.
  Keyword arguments:
  dm3_file (file) -- dm3 file object
  has_data (boolean) -- specifies if a given tag contains image data (True) or not (False)
  image_data (list) -- container for the image data
  Returns: None
  '''

  import struct

  ttype_header_size = 2 * type_size['i32']
  ttype_header = dm3_file.read(ttype_header_size)
  ttype_header = Reverse(ttype_header)

  ttype_items = { 'info_array_length': 0, 'info_array': [], 'data': [] }

  ttype_items['info_array_length'] = struct.unpack('i', ttype_header[:4])[0]
  info_array_size = ttype_items['info_array_length'] * type_size['i32']

  info_array_format = '>%di' % (ttype_items['info_array_length'])
  ttype_items['info_array'] = struct.unpack(info_array_format, dm3_file.read(info_array_size))

  type_id = 0
  array_size = 0
  data_size = 0

  # array
  if ttype_items['info_array'][0] == 20:    # array type id = 20
    array_size = ttype_items['info_array'][ttype_items['info_array_length'] - 1]

    # simple array
    if ttype_items['info_array_length'] == 3:      # length of simple array = 3
      type_id = ttype_items['info_array'][1]
      data_size = GetTypeSize(type_id) * array_size

    # array of groups
    elif ttype_items['info_array_length'] == 11:   # length of array of groups = 11
      type_id = ttype_items['info_array'][5]
      for i in range(0, ttype_items['info_array'][3]):
        data_size += GetTypeSize(ttype_items['info_array'][5 + 2 * i]) * array_size

  # not array
  else:
    # struct
    if ttype_items['info_array_length'] > 1:       # length of single entry = 1
      type_id = ttype_items['info_array'][4]
      for i in range(0, ttype_items['info_array'][2]):
        data_size += GetTypeSize(ttype_items['info_array'][4 + 2 * i])

    # single entry
    else:
      type_id = ttype_items['info_array'][0]
      data_size = GetTypeSize(ttype_items['info_array'][0])

  n_elements = data_size / GetTypeSize(type_id)
  ReadTagData(dm3_file, n_elements, type_id, tag_label)

#-----------------------------------------------------------------------------------

def ReadTagData(dm3_file, n_elements, type_id, tag_label):
  '''Reads data based on the information about datatypes and number of elements.
  Keyword arguments:
  dm3_file (file) -- dm3 file object
  n_elements (integer) -- number of elements (individual values) to be read
  type_id (integer) -- id number of the datatype given by dm3 format
  has_data (boolean) -- specifies if a given tag contains image data (True) or not (False)
  image_data (list) -- container for the image data
  Returns: None
  '''

  import struct

  n_elements = int(n_elements)

  data = []
  data_format = str(n_elements)
  type_size = GetTypeSize(type_id)

  if type_id == 2:
    data_format += type_format['i16']
  elif type_id == 4:
    data_format += type_format['ui16']
  elif type_id == 3:
    data_format += type_format['i32']
  elif type_id == 5:
    data_format += type_format['ui32']
  elif type_id == 6:
    data_format += type_format['float']
  elif type_id == 7:
    data_format += type_format['double']
  elif type_id == 8:
    data_format += type_format['bool']
  elif type_id == 9:
    data_format += type_format['char']
  elif type_id == 10:
    data_format += type_format['uchar']

  data = struct.unpack(data_format, dm3_file.read(n_elements * type_size))

  if tag_label == 'RestoreImageDisplayBounds':
    tags_data[tag_label] = [ int(dim) for dim in data[2:] ]
  elif tag_label == 'Scale' and tags_data['Units'] != 'nm':
    tags_data[tag_label] = [data[0] * 1e-9] * 2
  elif tag_label == 'Units':
    tags_data[tag_label] = ''.join([chr(i) for i in data])
  elif tag_label == 'Data':
    tags_data[tag_label] = data

#------------------------------------------------------------------------------------

def SaveDm3AsPng(image_data, image_dims, dm3_fname):
  '''Saves image data as png file.
  Image data is a matrix of integer values. Each value corresponds to a single greyscale pixel.
  Keyword arguments:
  image_data (list) -- data which contains information about pixel values
  dm3_fname (string) -- path of the dm3 file
  Returns: None
  '''

  import numpy as np
  from PIL import Image as im

  image1d = np.asarray(image_data)
  image2d = np.reshape(image1d, tuple(image_dims))

  image2d_rescaled = ((image2d - image2d.min()) * 255.0 / image2d.max()).astype(np.uint8)

  image = im.fromarray(image2d_rescaled)
  image.save(dm3_fname.replace('.dm3', '.png'))

#------------------------------------------------------------------------------------

# Run this block of code if a module is run as a standalone program
if __name__ == '__main__':
  ReadDm3File('img1.dm3')
