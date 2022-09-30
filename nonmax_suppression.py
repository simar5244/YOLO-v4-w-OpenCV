indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   
cv2.imshow("object detection", image)

cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)

cv2.destroyAllWindows()
