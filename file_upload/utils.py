ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension in ['csv', 'xlsx', 'xls', 'json', 'txt']:
        return 'data'
    elif extension == 'pdf':
        return 'pdf'
    elif extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']:
        return 'image'
    else:
        return 'unknown'