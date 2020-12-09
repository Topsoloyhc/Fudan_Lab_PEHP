def saveJson(person, fileindex, fileindexnew, oldBbox, newBbox):
    writepath, imagepath, rootpath = 'slice/crop/', 'slice/Images/', 'slice/Instances/'
    newjsondata = {}
    newjsondata['Input'] = {
        'Filename': "%07d" % fileindex,
        'Imagepath': imagepath + "%07d" % fileindex + '.jpg',
        'Instance_path': writepath + 'Instances/' + fileindexnew + '.png',
        'Human_ids_path': rootpath + 'Human_ids/' + "%07d" % fileindex + '.png',
        'Category_ids_path': rootpath + 'Category_ids/' + "%07d" % fileindex + '.png',
        'Instance_ids_path': rootpath + 'Instance_ids/' + "%07d" % fileindex + '.png',
    }
    newjsondata['Output'] = {
        'Filename': fileindexnew,
        'Imagepath': writepath + 'Images/' + fileindexnew + '.jpg',
        'Instance_path': writepath + 'Instances/' + fileindexnew + '.png',
        'Human_ids_path': writepath + 'Human_ids/' + fileindexnew + '.png',
        'Category_ids_path': writepath + 'Category_ids/' + fileindexnew + '.png',
        'Instance_ids_path': writepath + 'Instance_ids/' + fileindexnew + '.png',
    }

    newjsondata['Bbox'] = {
        'Orginal': oldBbox,
        'New': newBbox
    }
    newjsondata['OrginalCoordinate'] = {  # 在原图中的坐标位置
        'Xmin': newBbox[0],
        'Ymin': newBbox[1]
    }
    newjsondata['Resolution'] = [newBbox[2], newBbox[3]]

    newjsondata['DataTailor'] = {
        'Type': "Tailor_instance",
        'Original_Image': imagepath + "%07d" % fileindex + '.jpg',
        'Instance_Index': person
    }

    return newjsondata