from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(mask, ax, no_overlay=False):
    if no_overlay:
        ax.imshow(mask)
        return
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

class Renderer:
    def __init__(self) -> None:
        self.data = []
        pass

    def add_multiple(self, arrays: List[Dict]):
        for data in arrays:
            self.add(
                img=data.get('img', None),
                mask=data.get('mask', None),
                points=data.get('points', None),
                title=data.get('title', None),
            )
            pass
        pass

    def add(self, 
            img: np.ndarray, 
            mask: np.ndarray,
            points: Tuple[np.ndarray, np.ndarray], 
            title: str,
            ):
        
        assert img is not None or mask is not None, \
            'Both img or mask is None'
        
        _img = self._format_img(img)
        _mask = self._format_mask(mask)


        self.data.append(
            {
                'img': _img,
                'mask': _mask,
                'points': points,
                'title': title or 'Untitled'
            }
        )
        
        pass

    def _format_mask(self, mask):
        if mask is None: 
            return None

        assert mask.ndim <= 3, f"Mask dim <= 3, while get {mask.shape}"
        # Reshape mask to dim of 2
        if mask.ndim == 3:
            if mask.shape[0] == 1: 
                _mask = mask[0]
            elif mask.shape[-1] == 1: 
                _mask = mask[:, :, 0]
        else:
            _mask = mask.copy()

        return _mask

    def _format_img(self, img):
        if img is None: return None

        # Check for img data
        assert img.ndim < 4, "Out of control"
        _img = img.copy()
        if img.ndim == 2:
            _img = _img[:, :, None]

        if img.ndim == 3:
            assert _img.shape[-1] == 1 or _img.shape[-1] == 3, \
                f"Invalid shape {img.shape} transform to {_img.shape}"
            pass
        return _img

    def show_all(self, save_path: str=None):
        assert len(self.data) > 0, "There is no data to be rendered"
        f, ax, n_row, n_col = self._get_valid_subplot(len(self.data))

        for i1 in range(n_row):
            for i2 in range(n_col):
                idx = i1 * n_col + i2
                if idx >= len(self.data): break

                _data = self.data[idx]
                if _data['img'] is not None:
                    ax[i1, i2].imshow(_data['img'])
                    pass

                if _data['mask'] is not None:
                    overlay_mask(
                        _data['mask'], 
                        ax[i1, i2], 
                        no_overlay=_data['img'] is None
                        )
                    pass

                if _data['points'] is not None:
                    [pcoors, plabels] = _data['points']
                    self._render_point_label(
                        ax[i1, i2], pcoors, plabels, chosen_value=1, color_str='g')
                    self._render_point_label(
                        ax[i1, i2], pcoors, plabels, chosen_value=0, color_str='r')
                    pass

                ax[i1, i2].set_title(_data['title'])
                pass
        pass

        if save_path:
            f.savefig(save_path)
            plt.close()
            return
        else:
            f.show()
        pass
        

    def _render_point_label(self, ax, pcoors, plabels, chosen_value=1, color_str='g'):
        if pcoors is None: 
            return
        if plabels is None: 
            return
        msk = plabels == chosen_value
        if msk.any():
            xs = pcoors[msk][:, 0]
            ys = pcoors[msk][:, 1]
            ax.scatter(xs, ys, c=color_str)


    def _get_valid_subplot(self, n):
        if n <= 3:
            n_row, n_col = 1, n
        
        n_row = int(np.floor(np.sqrt(n)))
        n_row, n_col = n_row, n_row + 1
        f, axes = plt.subplots(n_row, n_col, squeeze=False)

        return f, axes, n_row, n_col
    
    def reset(self):
        self.data.clear()
        plt.close()