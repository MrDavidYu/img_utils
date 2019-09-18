import csv
import os
from glob import glob
import xml.etree.ElementTree as ET

from PIL import Image

"""
List of helper, enhancement and transformation funcs commonly used for prcessing
PascalVOC label XMLs.
During init, pass in both the (absolute) image dir and labels dir, making sure that
the filenames are in the format of <img_name>.png/jpg/tif and <img_name>.xml
The two dir can be the same.

Also supports additional voc util functions such as one-step conversions from other
types of dataformats.

List of methods for VOC class ----------------------------------------

get_bboxes():               Returns a list of lists for the image name supplied.
                            Each record = [class, xmin, ymin, xmax, ymax]
init_voc_datastruct:        Initializes _voc_datastruct
update_filename():          Updates the <filename>...</filename> tag elements with the
                            actual names of the file.
update_folder_name():       Updates the <folder>...</folder> tag elements with the
                            specified name.
get_bbox_count():           Returns number of bboxes for all files.


List of util functions: -----------------------------------------------

write_VOC_using_bbox:       Write to VOC format given bbox as a list of lists.

split_raster:               Splits input raster to output_file along with bboxes.
                            Padding is applied to ensure no hard cut-off of bboxes.

convert_vtrans_data         Converts VTrans right of way imagery "VOC" format into
                            usable VOC format. In-place operation.
"""

class VOCDataStructException(Exception):
    pass


class VOC:
    def __init__(self, images_path, labels_path):
        # TODO: currently, images_path doesn't do anything
        self.images_path = images_path
        self.labels_path = labels_path
        self._images = list()
        self._labels = list()
        self._img_suffix = ['*.png','*.jpg','*.tif']
        for suffix in self._img_suffix:
            self._images = glob(os.path.join(images_path, suffix))
            if len(self._images) != 0:
                break
        self._labels = glob(os.path.join(labels_path, '*.xml'))
        assert len(self._images) == len(self._labels)

        """
        self._voc_storage:
        | ...                                              |
        | {ImgName: [[class, xmin, ymin, xmax, ymax],...]} |
        | ...                                              |
        """
        self._voc_datastruct = dict()
        self._is_voc_datastruct_created = False

    
    def init_voc_datastruct(self):
        """ Initialize self._voc_datastruct which is the struct
        that contains all label files read in.
        """
        for label in self._labels:
            filename = label.split('/')[-1][:-4]
            tree = ET.parse(label)
            root = tree.getroot()
            record_bboxes = list()
            for child in root:
                if child.tag == 'object':
                    record_bbox = list()
                    class_ = None
                    xmin = None
                    ymin = None
                    xmax = None
                    ymax = None
                    for child2 in child:
                        if child2.tag == 'name':
                            class_ = child2.text
                        if child2.tag == 'bndbox':
                            for child3 in child2:
                                if child3.tag == 'xmin':
                                    xmin = child3.text
                                if child3.tag == 'ymin':
                                    ymin = child3.text
                                if child3.tag == 'xmax':
                                    xmax = child3.text
                                if child3.tag == 'ymax':
                                    ymax = child3.text
                    record_bbox.append(class_)
                    record_bbox.append(xmin)
                    record_bbox.append(ymin)
                    record_bbox.append(xmax)
                    record_bbox.append(ymax)
                    record_bboxes.append(record_bbox)
            self._voc_datastruct[filename] = record_bboxes
        self._is_voc_datastruct_created = True
        return


    def get_bboxes(self, img_name):
        """
        Returns a list of lists for the image name supplied.
        Each record = [class, xmin, ymin, xmax, ymax]
        Input:  img_name: Filename which doesn't include path and
                doesn't include suffices e.g. "an_img.jpg" --> "an_img"
        """
        if not self._is_voc_datastruct_created:
            raise VOCDataStructException(
                "VOC Datastruct not generated. Call init_voc_datastruct() first")
            exit(1)
        
        return self._voc_datastruct[img_name]
    
    
    def get_bbox_count(self):
        """ Returns number of bboxes for all files"""
        count = 0
        for _, bboxes in self._voc_datastruct.items():
            count += len(bboxes)
        return count


def update_folder_name(filename, folder_name):
    """ Update the <folder>...</folder> text with folder_name
    """
    # with is like your try .. finally block in this case
    with open(filename, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # now change the 2nd line, note that you have to add a newline
    data[1] = "\t<filename>"+folder_name+"</filename>\n"

    # and write everything back
    with open(filename, 'w') as file:
        file.writelines(data)


def write_VOC_using_bbox(filename, output_file, width, height, bboxes):
    """ Write to VOC format given bbox as a list of lists.
    Where each record is [xmin, ymin, xmax, ymax, classname]
    Filename is something like test.jpg
    output_file is something like /home/test.xml
    """
    with open(output_file, 'w') as outfile:
        outfile.write("<annotation>\n")
        outfile.write("\t<folder>VOC2012</folder>\n")
        outfile.write("\t<filename>"+filename+"</filename>\n")
        outfile.write("\t<path>"+output_file[:-4]+".jpg"+"</path>\n")
        outfile.write("\t<source>\n")
        outfile.write("\t\t<database>Unknown</database>\n")
        outfile.write("\t</source>\n")
        outfile.write("\t<size>\n")
        outfile.write("\t\t<width>"+str(width)+"</width>\n")
        outfile.write("\t\t<height>"+str(height)+"</height>\n")
        outfile.write("\t\t<depth>3</depth>\n")
        outfile.write("\t</size>\n")
        outfile.write("\t<segmented>0</segmented>\n")

        for bbox in bboxes:

            xmin = bbox[1]
            ymin = bbox[2]
            xmax = bbox[3]
            ymax = bbox[4]
            class_ = bbox[0]

            outfile.write("\t<object>\n")
            outfile.write("\t\t<name>"+class_+"</name>\n")
            outfile.write("\t\t<pose>Unspecified</pose>\n")
            outfile.write("\t\t<truncated>0</truncated>\n")
            outfile.write("\t\t<difficult>0</difficult>\n")
            outfile.write("\t\t<bndbox>\n")
            outfile.write("\t\t\t<xmin>"+str(int(xmin))+"</xmin>\n")
            outfile.write("\t\t\t<ymin>"+str(int(ymin))+"</ymin>\n")
            outfile.write("\t\t\t<xmax>"+str(int(xmax))+"</xmax>\n")
            outfile.write("\t\t\t<ymax>"+str(int(ymax))+"</ymax>\n")
            outfile.write("\t\t</bndbox>\n")
            outfile.write("\t</object>\n")

        outfile.write("</annotation>\n")
    return

def split_raster(raster, output_file, output_file_xml, width, height,
                 bboxes, x_split_factor, y_split_factor):
    """ Splits input raster to output_file.
    Width and height are w.r.t. the original file.
    Bboxes is a list of [xmin, ymin, xmax, ymax, classname]
    x and y split factors are the number of output raster pieces
    in the x and y directions respectively.
    
    N.B. if there are remainder image segments in the x or y
    directions after the split, they will NOT be kept. However these
    should only be a couple of pixels.
    
    For e.g. output_file is </foo/test_file.jpg>, x_split_factor
    is 3 and y_split_factor is 2. Then the following outputs
    would be produced:
    </foo/test_file_x1_y1.jpg>, <..._x2_y1.jpg>, <..._x3_y1.jpg>
    </foo/test_file_x1_y2.jpg>, <..._x2_y2.jpg>, <..._x3_y2.jpg>
    
    Returns 3 lists: split_rasters, a list of appropriate filenames
    for these rasters, and a list of lists wherin each list contains
    the bounding boxes for that split raster.
    And 2 ints: chipped image width and height
    """
    split_rasters = list()
    split_rasters_name = list()
    split_rasters_xml_name = list()
    split_bboxes = list()
    
    x_split_width = int(width/x_split_factor)
    y_split_height = int(height/y_split_factor)
    
    image = Image.open(raster)
    
    curr_y = 0
    for y in range(y_split_factor):
        curr_x = 0
        for x in range(x_split_factor):
            bboxes_for_curr_crop = list()
            cropped_img = image.crop((curr_x, curr_y,
                                     curr_x + x_split_width,
                                     curr_y + y_split_height))
            new_raster_name = (output_file[:-4] + "_x" + str(x+1)
                               + "_y" + str(y+1) + output_file[-4:])
            new_xml_name = (output_file_xml[:-4] + "_x" + str(x+1)
                            + "_y" + str(y+1) + output_file_xml[-4:])
            for bbox in bboxes:
                new_bbox = list()
                if ((((int(bbox[1]) >= curr_x) and (int(bbox[1]) < (curr_x + x_split_width)))
                       and ((int(bbox[2]) >= curr_y) and (int(bbox[2]) < (curr_y + y_split_height))))
                       or (((int(bbox[3]) > curr_x) and (int(bbox[3]) <= (curr_x + x_split_width)))
                       and ((int(bbox[4]) > curr_y) and (int(bbox[4]) <= (curr_y + y_split_height))))):
                    
                    new_bbox.append(bbox[0])
                    # since new bbox may exist in a chip that has a
                    # relative origin of (curr_x, curr_y), new bbox
                    # coords must also be ammended:
                    new_bbox.append(max((int(bbox[1]) - curr_x), 0))
                    new_bbox.append(max((int(bbox[2]) - curr_y), 0))
                    new_bbox.append(min((int(bbox[3]) - curr_x), x_split_width))
                    new_bbox.append(min((int(bbox[4]) - curr_y), y_split_height))
                    # TODO: Add additional statement to check wwhether the slice is too small
                    bboxes_for_curr_crop.append(new_bbox)
            curr_x += x_split_width
            split_rasters.append(cropped_img)
            split_rasters_name.append(new_raster_name)
            split_rasters_xml_name.append(new_xml_name)
            split_bboxes.append(bboxes_for_curr_crop)
        curr_y += y_split_height

    return split_rasters, split_rasters_name, split_rasters_xml_name, split_bboxes, x_split_width, y_split_height


def convert_vtrans_data(labels_path, coord_path=None):
    """
    Input:  labels_path: Directory. Path to VOC xml data. This is also the output location.
                         Pass in absolute address.
            coord_file:  File to store output coordinates. Output will be in a csv format akin
                         to the ADOT Photolog DB csv format. e.g. /your/path/foo.csv
                         Pass in absolute address.

    Sample contents of coord_file:
    OBJECTID,SHAPE,SessionID,ImageCount,ImageDate,GpsTime,Longitude,Latitude,Altitude,HDOP,GpsSpeed,Bearing
    1231915,,215,00027,26-Sep-96,26-Sep-96,-112.6328972,33.86396338,598,0.9,26.21,43.28
    ...


    <""> will be insterted in case where no data is available.
    
    N.B. VTrans dataset also includes altitude. This information is included in the output to coord_path.
    However because no bearing information in included in the metadata, lat/long inference is currently
    not possible with VTrans dataset.
    
    TODO: Currently lat long and alt are not parsed from input file
    """
    xml_files = glob(os.path.join(labels_path, "*.xml"))
    is_first_file = True  # used to check whether field names should be written to csv in coord_path
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        record_bboxes = list()
        filename = None
        width = None
        height = None
        lat = None
        long = None
        alt = None
        for child in root:
            if child.tag == 'filename':
                filename = child.text
            if child.tag == 'size':
                for child2 in child:
                    if child2.tag == 'width':
                        width = child2.text
                    if child2.tag == 'height':
                        height = child2.text
            if child.tag == 'object':
                record_bbox = list()
                class_ = None
                xmin = None
                ymin = None
                xmax = None
                ymax = None
                for child2 in child:
                    if child2.tag == 'name':
                        class_ = child2.text
                    if child2.tag == 'bndbox':
                        for child3 in child2:
                            if child3.tag == 'xmin':
                                xmin = child3.text
                            if child3.tag == 'ymin':
                                ymin = child3.text
                            if child3.tag == 'xmax':
                                xmax = child3.text
                            if child3.tag == 'ymax':
                                ymax = child3.text
                record_bbox.append(class_)
                record_bbox.append(xmin)
                record_bbox.append(ymin)
                record_bbox.append(xmax)
                record_bbox.append(ymax)
                record_bboxes.append(record_bbox)

        # Now write parsed input to output
        write_VOC_using_bbox(filename, xml_file, width, height, record_bboxes)
        
        if coord_path is not None:
            # Write in append mode
            with open(coord_path, 'a') as writeFile:
                writer = csv.writer(writeFile)
                lines = list()
                if is_first_file:
                    lines.append(["OBJECTID","SHAPE","SessionID","ImageCount","ImageDate","GpsTime",
                                  "Longitude","Latitude","Altitude","HDOP","GpsSpeed","Bearing"])
                    is_first_file = False
                lines.append([filename.split('.')[0], "", "", "", "", "",
                             long, lat, alt, "", "", ""])
                writer.writerows(lines)
