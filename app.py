import cv2,numpy as np,face_recognition,os

# ImageDb
path='./Faces_Images'

# Globales variables 
image_list=[] #list of images
name_list=[] #list of names

# Gral all images from the folder
mylist=os.listdir(path)

# Load the images
for img in mylist:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png','.jpeg']:
        curImg=cv2.imread(os.path.join(path,img))
        image_list.append(curImg)
        imgName=os.path.splitext(img)[0]
        name_list.append(imgName)
    
# Define the face detection  and extraction features
def findEncodings(image_list,ImgName_list):
    """_summary_
    Define the face detection and extraction features
    
    Args:
        img_list (list): list of BGR of images
        ImgName_list (list) : List of image  names """
        
    signature_db = []
    count = 1
    for myImg, name in zip(image_list,ImgName_list):
        img=cv2.cvtColor(myImg,cv2.COLOR_BGR2RGB)
        signature=face_recognition.face_encodings(img)[0]
        signature_class= signature.tolist()+[name]
        signature_db.append(signature_class)
        print(f'{int((count/len(image_list))*100)} % extracted')
        count+=1
    signature_db=np.array(signature_db)
    np.save('.\Face_recognition_azure\Face_recognition_api\Face_Signatures_db.npy',signature_db)
    print(f"Signature_db stored")
    
def main():
    findEncodings(image_list,name_list)
 
if __name__ == "__main__":
    main()



