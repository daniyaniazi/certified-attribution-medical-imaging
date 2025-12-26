import sys
sys.path.append(r'd:\git projects\certified-attribution-medical-imaging')

try:
    from src.models.factory import get_model
    m,c = get_model('resnet18', num_classes=2, pretrained=False, device='cpu')
    print('Imported OK:', type(m), c.backbone)
    m2,c2 = get_model('mobilenet_v2', num_classes=3, pretrained=False, device='cpu')
    print('MobileNet OK:', type(m2), c2.backbone)
except Exception as e:
    print('ERROR:', e)
    raise
