# yolo-from-scratch

An educational implementation of the YOLO (You Only Look Once) object detection algorithm using **TensorFlow + Keras**.  
The goal is to understand *how YOLO works under the hood* by building each component step-by-step, starting with YOLOv1 and extending to YOLOv2.

---

## 📌 Project Goals
- Learn object detection fundamentals by re-implementing YOLO.
- Build a modular TensorFlow/Keras training pipeline:
  - Data loading & preprocessing
  - Target assignment (grid / anchors)
  - Backbone + detection head
  - Custom composite loss
  - Inference with decoding + NMS
  - Evaluation (mAP) and visualization
- Extend from **YOLOv1 → YOLOv2 (YOLO9000)** with anchors, ignore regions, and multi-scale training.

---

## 🗺 Roadmap

### Phase 1: YOLOv1
1. **Dataset & Pipeline**
   - Use Pascal VOC or a small custom dataset.
   - Preprocessing: resize, normalize, augment.
   - Convert labels to YOLO grid format (`S×S×(B*5 + C)`).

2. **Model**
   - Lightweight ConvNet backbone (Conv + BN + LeakyReLU).
   - Detection head with final Conv2D producing `S×S×(B*5 + C)`.

3. **Loss**
   - Box regression (MSE on (x,y), √w, √h).
   - Objectness (positive & negative cells).
   - Class prediction loss (CE/MSE).
   - Tunable weights (λ_coord, λ_noobj).

4. **Training**
   - `model.fit` or custom training loop.
   - Warmup + LR schedule.
   - Monitor separate loss terms.

5. **Inference**
   - Decode predictions to absolute boxes.
   - Confidence = objectness × class probability.
   - Apply NMS for final predictions.

6. **Evaluation & Visualization**
   - Compute **mAP@[0.5]**.
   - Visualize predicted vs ground truth boxes.

---

### Phase 2: YOLOv2 (YOLO9000)
1. Introduce **anchors** (k-means or VOC defaults).
2. Update target assignment (anchor responsibility + ignore IoU threshold).
3. Loss modifications for anchor parameterization (t_x, t_y, t_w, t_h).
4. Add **multi-scale training** (input sizes 320–608).
5. Strengthen backbone.

---

## ✅ Deliverables
- **Data pipeline** (`tf.data`)
- **YOLOv1 model** (Keras Sequential/Functional API)
- **Custom loss** (composite with masks)
- **Inference decoder + NMS**
- **Training script** (with callbacks, checkpoints, LR schedule)
- **Evaluation script** (mAP, PR curves)
- **Visualization tools** (targets, predictions)

---

## 📅 Suggested Milestones
- **Week 1:** Dataset & targets verified visually.
- **Week 2:** Loss stable, short overfit run on small dataset.
- **Week 3:** End-to-end YOLOv1 training on subset, sanity mAP.
- **Week 4:** Full YOLOv1 training; visualization + metrics.
- **Phase 2:** Add anchors → YOLOv2 features.

---

## ⚠️ Notes & Pitfalls
- Verify target assignment visually before training.
- Loss balancing is critical; tune λ_noobj carefully.
- Monitor box/obj/cls loss separately to catch imbalances.
- Ensure augmentations update boxes consistently.

---

## 📂 Progress





