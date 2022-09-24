# PixJS

This repository contains the code for our paper on [PixJS: A novel chaos-based approach for image encryption](https://doi.org/10.1002/cpe.6990). 

## Abstract
Image encryption has variety of applications in the healthcare, military, e-commerce, and education sectors. Along with processing images for various applications, secure storage of such images is essential. For many years, encryption of images is a fruitful method for the secure transmission of multimedia data that carries sensitive information across multiple mediums. There is umpteen number of image encryption methods available which exhibit improved performance either in time, space, or susceptance to attacks in a controlled environment. Few image encryption algorithms have higher computational complexity, whereas some are immune to a handful of attacks (either differential, statistical, brute force, or known-plaintext and chosen-plain text attacks). We have developed a novel chaos-based image encryption algorithm—PixJS for 8-bit grayscale images based on linear feedback shift register, logistic map, and jumbling process. The algorithm was initially proposed for the text-based passwords called as jumbling-salting algorithm in 2014. We conducted a comprehensive analysis of the proposed algorithm on sets of symmetric and asymmetric images (10 samples each) and addressed four research questions primarily based on speed and security parameters. After conducting thorough experimentation, we found that our algorithm outperformed existing algorithms in security and has proven to be resistant to statistical, differential, brute-force, known plain text, and chosen plain text attacks.


If you use any of our code, please cite:

```
@article{tuli2022pixjs,
  title={PixJS: A novel chaos-based approach for image encryption},
  author={Tuli, Rohan and Soneji, Hitesh and Vahora, Sahil and Churi, Prathamesh and Bangalore, Nagachetan M},
  journal={Concurrency and Computation: Practice and Experience},
  pages={e6990},
  year={2022},
  publisher={Wiley Online Library}
}
```

## License
This project is licensed under the GNU General Public License v3.0. See LICENSE for more details.
