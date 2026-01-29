"""Funzioni di utilità per calcoli su bounding box."""

def get_center_of_bbox(bbox):
    """Calcola il centro del bbox."""
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    """Calcola la larghezza del bbox."""
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    """Calcola la distanza euclidea tra due punti."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    """Calcola le distanze separate x e y tra due punti."""
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    """Calcola la posizione del punto centrale inferiore del bbox."""
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)