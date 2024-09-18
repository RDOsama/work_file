from flask import Flask, request, send_file, jsonify, make_response
from io import BytesIO
import io
from ultralytics import YOLO
from PIL import Image 
from torchvision import transforms
import numpy as np
import cv2
import torch
import base64
import cvzone
import pdb
import pandas as pd

app = Flask(__name__)

# Load the YOLO model
model = YOLO('best_V3.pt')

# Constants
BOX_CONF_THRESH = 0.5
BOX_IOU_THRESH = 0.5
KPT_CONF_THRESH = 0.5
inc = 15

ann_meta_data = pd.read_csv("animal_pose_traning/keypoint_definitions.csv")
COLORS = ann_meta_data["Hex colour"].values.tolist()

COLORS_RGB_MAP = []
for color in COLORS:
    R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    COLORS_RGB_MAP.append({color: (R,G,B)})

def resize_with_aspect_ratio(img, new_width=None, new_height=None):
    # Get the current height and width
    height, width = img.shape[:2]

    # If only width is specified
    if new_width is not None and new_height is None:
        # Calculate the aspect ratio and new height
        aspect_ratio = width / height
        new_height = int(new_width / aspect_ratio)

    # If only height is specified
    elif new_height is not None and new_width is None:
        # Calculate the aspect ratio and new width
        aspect_ratio = height / width
        new_width = int(new_height / aspect_ratio)

    # If both width and height are specified, ignore aspect ratio
    elif new_width is not None and new_height is not None:
        pass

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))
    # resized_img =     cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return resized_img
# def draw_landmarks(image, landmarks):

#     radius = 5
#     # Check if image width is greater than 1000 px.
#     # To improve visualization.
#     if (image.shape[1] > 1000):
#         radius = 8

#     for idx, kpt_data in enumerate(landmarks):

#         loc_x, loc_y = kpt_data[:2].astype("int").tolist()
#         color_id = list(COLORS_RGB_MAP[int(kpt_data[-1])].values())[0]

#         cv2.circle(image,
#                    (loc_x, loc_y),
#                    radius,
#                    color=color_id[::-1],
#                    thickness=-1,
#                    lineType=cv2.LINE_AA)

#     return image

def draw_landmarks(image, landmarks):
    radius = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color for text
    thickness = 1

    # Check if image width is greater than 1000 px.
    # To improve visualization.
    if image.shape[1] > 1000:
        radius = 8

    for kpt_data in landmarks:
        loc_x, loc_y = kpt_data[:2]
        color_id = list(COLORS_RGB_MAP[int(kpt_data[-1])].values())[0]

        # Draw a circle for the landmark
        cv2.circle(image,
                   (loc_x, loc_y),
                   radius,
                   color=color_id[::-1],
                   thickness=-1,
                   lineType=cv2.LINE_AA)

        # Add point number as text label above the landmark
        cv2.putText(image, str(kpt_data), (loc_x, loc_y - 10),
                    font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

    return image

def wear_cap(cap_img,animal_image,  filter_kpts):
    ############################### cap working ############################### 
    
    print('Shape of animal_image:', animal_image.shape)
    print('Data type of animal_image:', animal_image.dtype)

    ################# Check two points (if 14 and 15 exist in the third item of each list)
    
    point_14_exists = any(14 == sublist[2] for sublist in filter_kpts)
    point_15_exists = any(15 == sublist[2] for sublist in filter_kpts)


    

    ################ check four points 

    # Points to check
    points_to_check = [14, 15, 19, 18]
    
    # Check if all points exist in the third item of any list
    all_points_exist = all(point in [sublist[2] for sublist in filter_kpts] for point in points_to_check)
      
    if all_points_exist:
        # print("************ All points exist.")
        print(f'----------****** four points exits ******----------{all_points_exist}\n')

        

        ## *********** check if 19 > 15 and 18 > 14 (condition to correct image flip )
        # get value of 19 and 15
        point_19 = next(sublist for sublist in filter_kpts if sublist[2] == 19)
        point_15 = next(sublist for sublist in filter_kpts if sublist[2] == 15)
        # get value of 18 and 14
        point_18 = next(sublist for sublist in filter_kpts if sublist[2] == 18)
        point_14 = next(sublist for sublist in filter_kpts if sublist[2] == 14)

        if point_19[1] > point_15[1]:
            # Calculate the width between points 14 and 15
            width_19_15 = np.sqrt((point_19[0] - point_15[0])**2 + (point_19[1] - point_15[1])**2)
            
            # Calculate the width between points 14 and 15
            width_18_14 = np.sqrt((point_18[0] - point_14[0])**2 + (point_18[1] - point_14[1])**2)
            
            # print('>>>>>>>>>>>>>>>> length of 18 and 14: ', width_18_14)
            # print('>>>>>>>>>>>>>>>> length of 19 and 15: ', width_19_15)
            # print('\n19, 15, 18, 14 origranl', [point_19, point_15, point_18, point_14])
    
            point_19[1] = int(point_19[1] - 2 * width_19_15)
            point_18[1] = int(point_18[1] - 2 * width_18_14)
            
            # print('19, 15, 18, 14 decreamented\n', [point_19, point_15, point_18, point_14])

            filter_kpts[-1] = point_19 
            filter_kpts[-2] = point_18
        else:
            pass
            
        cap = cap_img
        cap_h,cap_w =  cap.shape[:2]

        print(f'******** new filter_kpts: {filter_kpts}\n', )
        width_14_15 = np.sqrt((point_14[0] - point_15[0])**2 + (point_14[1] - point_15[1])**2)
        width_19_15 = np.sqrt((point_15[0] - point_19[0])**2 + (point_15[1] - point_19[1])**2)

        head_coordinates = []
        for item in filter_kpts:
            
            if item[2] in points_to_check:
                if item[2] == 14:
                    # print('before ####### item[2]item[2]item[2]',item)
                    item[0] = item[0] + int(width_14_15/4)
                    # print('after ####### item[2]item[2]item[2]',int(item[0] - (width_14_15/4)))
                    
                    head_coordinates.append(item)
                elif item[2] == 15:
                    # print('####### item[2]item[2]item[2]',item)
                    item[0] = item[0] - int(width_14_15/4)
                    head_coordinates.append(item)
                else:
                    # print('else  ####### item[2]item[2]item[2', item[2])
                    item[1] = item[1] - int(width_19_15/6)
                    
                    head_coordinates.append(item)

        # print('******** new filter_kpts: ', head_coordinates)

        # head_coordinates = [item for item in filter_kpts if item[2] in points_to_check]
        head_coordinates_position = sorted(head_coordinates, key=lambda x: x[2], reverse=True)
        print('-------------------- head coordinates position with keypoint numbers: ', head_coordinates_position)
        # print('-------------------- length of head coordinates position: ', len(head_coordinates_position))
        
        head_coordinates_position = [[item[0], item[1] + inc] for item in head_coordinates_position]
        # print('-------------------- head coordinates position without keypoint numbers: ', head_coordinates_position)
        
        pts1=np.float32([[0,0],[cap_w,0],[0,cap_h],[cap_w,cap_h]])
        pts2=np.float32(head_coordinates_position)
        
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    
        height, width, channels = animal_image.shape
        im1Reg = cv2.warpPerspective(cap_img, h, (width, height))
        # animal_image = cv2.cvtColor(animal_image, cv2.COLOR_BGR2RGB) 
        print('animal_image shape', animal_image.shape)
        print('im1Reg shape', im1Reg.shape)


        animal_image_result = cvzone.overlayPNG(animal_image, im1Reg, (0, 0))
        
        # cv2.imshow("Original Image", animal_image_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return animal_image_result, 'image transforms successfully'
    
    elif point_14_exists and point_15_exists:

        
        # get value of 14 and 15
        point_14 = next(sublist for sublist in filter_kpts if sublist[2] == 14)
        point_15 = next(sublist for sublist in filter_kpts if sublist[2] == 15)
        point_19 = next(sublist for sublist in filter_kpts if sublist[2] == 19)
        point_18 = next(sublist for sublist in filter_kpts if sublist[2] == 18)

        # print('################# point_14: ', point_14)
        # print('################# point_15: ', point_15)

        
        # print("********************* Both 14 and 15 exist.")
        

        
        # Calculate the width between points 14 and 15
        width_14_15 = np.sqrt((point_14[0] - point_15[0])**2 + (point_14[1] - point_15[1])**2)
        width_18_19 = np.sqrt((point_18[0] - point_19[0])**2 + (point_18[1] - point_19[1])**2)
        width_15_19 = np.sqrt((point_15[0] - point_19[0])**2 + (point_19[1] - point_19[1])**2)


        # print('################# width_14_15: ', width_14_15)
        # print(f"********************* The width between points 14 and 15 is: {width_14_15}")

        resized_accessories_img  = resize_with_aspect_ratio(cap_img, int(width_18_19))

        
        # animal_image = cv2.cvtColor(animal_image, cv2.COLOR_RGB2BGR) 
        # print('%%%%%%%%%%% shape of png', resized_accessories_img.shape)
        

        percentage_to_subtract = 38 
        offset_y = int(percentage_to_subtract / 100 * resized_accessories_img.shape[1])

        ##### animal_image_result = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_15[0] - int(width_14_15/6), point_15[1] - offset_y ))
        print('\n\npoint_15[0] - int(width_14_15/4)', point_15[0] - int(width_14_15/4))
        print('point_15[1] - offset_y ', point_15[1] - offset_y )
        # if point_19[1] > point_15[1]:
        #     # print('\n\nreater thanaaanananananananananan')
        #     # # pass
        #     # print('width_14_15: ',  width_14_15 )
        #     # print('width_15_19: ',  width_15_19 )
        #     # print('point_15[1] - int(width_15_19)', point_15[1] - int(width_15_19))
        #     # print('resized_accessories_img.shape[1]', resized_accessories_img.shape[1])
        #     # print('offset_y: ', offset_y)
        #     animal_image_result = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_15[0] , int(width_14_15/4), point_15[1] - offset_y ))

        #     return animal_image_result,  'image transforms successfully'
        # else:
        #     animal_image_result = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_15[0] - int(width_14_15/4), point_15[1] - offset_y ))

        #     return animal_image_result,  'image transforms successfully'
        
        if point_19[1] > point_15[1]:
            print('diffffffffffffffffffffffff', point_19[1] - point_15[1])
            animal_image_result = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_15[0] - int(width_14_15/4), point_19[1] - point_15[1]  ))

            return animal_image_result,  'image transforms successfully'
        else:

            animal_image_result = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_15[0] - int(width_14_15/4), point_15[1] - offset_y ))

            return animal_image_result,  'image transforms successfully'
    ################# Check two points code end
    



    else:
        missing_points = [point for point in points_to_check if point not in [sublist[2] for sublist in filter_kpts]]
        print(f" ************The following points are missing: {missing_points}")

        return animal_image, 'points not detected'
    ############################### cap working end ###############################


# def wear_collar(accessories_img ,animal_image,  filter_kpts):
#     ############################### collar working ############################### 
#    return animal_image

def wear_collar(accessories_img ,animal_image,  filter_kpts):
    ############################### collar working ############################### 
    # get value of 8 and 2
    point_8 = next(sublist for sublist in filter_kpts if sublist[2] == 8)
    point_2 = next(sublist for sublist in filter_kpts if sublist[2] == 2)
    # print('point_8 and point_2', point_8, point_2 )

    ### check if point 8 and point 2 exits 
    if point_8 and point_2:
        # print('exits+++++++++++++++++')

        #### width between 8 and 2
        width_8_2 = np.sqrt((point_2[0] - point_8[0])**2 + (point_2[1] - point_8[1])**2)

        #### resize image according to width if the 8 and 2 points 
        resized_accessories_img  = resize_with_aspect_ratio(accessories_img, int(width_8_2))

        # Calculate the midpoint
        midpoint_x = int((point_2[0] + point_8[0]) / 2)
        midpoint_y = int((point_2[1] + point_8[1]) / 2)
        # print("&&&&&&&& mid between 8 and 2 midpoint_x  and midpoint_y", midpoint_x, midpoint_y)

        
        point_17 = next(sublist for sublist in filter_kpts if sublist[2] == 17)
        
        mid_17_bet_width_8_2 = np.sqrt((midpoint_x - point_17[0])**2 + (midpoint_y - point_17[1])**2)
        # print('width between mid and 17: ', mid_17_bet_width_8_2)

        if mid_17_bet_width_8_2 <= 100:
            percentage_to_subtract = 35
            offset_y = int(percentage_to_subtract / 100 * resized_accessories_img.shape[1])
            image = cvzone.overlayPNG(animal_image, resized_accessories_img, (point_8[0] , point_8[1] - offset_y))
            
        else:
            midpoint_x_17 = int((midpoint_x + point_17[0]) / 2)
            midpoint_y_17 = int((midpoint_y + point_17[1]) / 2)
            # print("^^^^^^^^^^^^^^^^ mid between mid and 7 ", midpoint_x_17, midpoint_y_17)
            
            ######## show point in specific location
            percentage_to_subtract = 45
            offset_y = int(percentage_to_subtract / 100 * resized_accessories_img.shape[1])
            
            image = cvzone.overlayPNG(animal_image, resized_accessories_img, (midpoint_x_17 - offset_y, midpoint_y_17 - offset_y))
        
        return image, 'image transforms successfully'
        
        
    else:
        missing_points = [point for point in points_to_check if point not in [sublist[2] for sublist in filter_kpts]]
        print(f" ************The following points are missing: {missing_points}")

        return animal_image, 'points not detected'
    ############################## collar working end ###############################



@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if files are included in the request
        if 'ani_img' not in request.files or 'catag_img' not in request.files and 'catag_name' not in request.form:
            return jsonify({'error': 'Both files are required'})
        # pdb.set_trace()
        file1 = request.files['ani_img']
        file2 = request.files['catag_img']
        catag_name = request.form.get('catag_name')
        print('catag_name------------------ ', catag_name)

        # # Check if the files have valid content types (MIME types)
        # if not file1 or not file1.content_type.startswith('image/') or not file2 or not file2.content_type.startswith('image/'):
        #     return jsonify({'error': 'Invalid image files'})


        acc_img_np = np.fromstring(file2.read(), np.uint8)
        acc_img = cv2.imdecode(acc_img_np, cv2.IMREAD_UNCHANGED)
        # acc_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2RGB)

        # Read the image files using OpenCV
        ani_img_np = np.fromstring(file1.read(), np.uint8)
        ani_img = cv2.imdecode(ani_img_np, cv2.IMREAD_COLOR)
        # ani_img = cv2.cvtColor(ani_img, cv2.COLOR_BGR2RGB)

        ###################################### return byte image

        # # Resize the image to meet YOLO model requirements
        # ani_img = cv2.cvtColor(ani_img, cv2.COLOR_BGR2RGB)

        
        
        # Perform YOLO predictions
        results = model.predict(ani_img, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH)[0].cpu()
        # # Check if results are available
        # print(results)
        if results is None or results == []:
            return jsonify({'error': 'No predictions please enter correct image'})
        
        else:
            try:
                # Get the predicted boxes, conf scores and keypoints.
                # pred_boxes = results.boxes.xyxy.numpy()
                # pred_box_conf = results.boxes.conf.numpy()
                # pdb.set_trace()
                pred_kpts_xy = results.keypoints.xy.numpy()
                pred_kpts_conf = results.keypoints.conf.numpy()
                # print('\n pred_kpts_xy: ', pred_kpts_xy)
                # print('\n pred_kpts_conf: ', pred_kpts_conf, '\n\n')

                filter_kpts = []
                # Draw predicted bounding boxes, conf scores and keypoints on image.
                for kpts, confs in zip(pred_kpts_xy, pred_kpts_conf):
                    kpts_ids = np.where(confs > KPT_CONF_THRESH)[0]
                    filter_kpts = kpts[kpts_ids]
                    filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)

                    # filter_kpts = filter_kpts.astype("int").tolist()
                    
                    filter_kpts = [[int(x) for x in inner_list] for inner_list in filter_kpts]
                    print('\n----------------------- filter_kpts: ', filter_kpts)
                    # print('******** old filter_kpts: ', filter_kpts)
                # pdb.set_trace()
                if catag_name == "collarbelt":
                    res_img, response = wear_collar(acc_img, ani_img, filter_kpts)
                    
                    # ani_img = cv2.cvtColor(ani_img, cv2.COLOR_BGR2RGB)
                    # res_img = wear_collar(acc_img, ani_img, filter_kpts)
                    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    res_img = draw_landmarks(res_img, filter_kpts)

                elif catag_name == "cap":
                    res_img, response = wear_cap(acc_img, ani_img, filter_kpts)
                    # ani_img = cv2.cvtColor(ani_img, cv2.COLOR_RGB2BGR)
                    # res_img = wear_cap(acc_img, ani_img, filter_kpts)[0]
                    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    res_img = draw_landmarks(res_img, filter_kpts)
                    # cv2.imshow('check', res_img)
                    # waits for user to press any key
                    # (this is necessary to avoid Python kernel form crashing)
                    cv2.waitKey(0)

                    # closing all open windows
                    cv2.destroyAllWindows()
                    # print('Shape of res_img:', res_img.shape)
                    # print('Data type of res_img:', res_img.dtype)
                    # print('Content of res_img:', res_img)
                else:
                    print('Please give catagory name')
                    return jsonify({'error': 'Please give catagory name'})

                
                # # Convert the image to base64
                _, img_encoded = cv2.imencode('.png', cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                # # image_stream = BytesIO(img_base64)

                # # # Open the image using Pillow (PIL)
                # # image = Image.open(image_stream)

                # # Include the base64-encoded image in the response
                # response_data = {'response': response , 'image': img_base64}
                # # response_data = {'response': response, 'image': image }

                # return jsonify(response_data)
                # Decode the Base64 string back to binary data
                img_binary = base64.b64decode(img_base64)
                
                # Create a BytesIO stream from the binary data
                img_stream = io.BytesIO(img_binary)
                
                # Send the image file with the appropriate MIME type
                return send_file(img_stream, mimetype='image/png')
                
            except Exception as e:
                print('---------------- prediction error ', e)
                return jsonify({'error': "please enter correct animal image"})


    except Exception as e:
        print('---------------- error ', e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 6666)
