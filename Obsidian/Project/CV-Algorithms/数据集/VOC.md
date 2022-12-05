## 标注格式
```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000005.jpg</filename>  # 对应图片文件名
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>325991873</flickrid>
	</source>
	<owner>
		<flickrid>archintent louisville</flickrid>
		<name>?</name>
	</owner>
	<size>  # 图像原始尺寸
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>  # 是否用于分割
	<object>
		<name>chair</name>  # 物体类别
		<pose>Rear</pose>  # 拍摄角度：front, rear, left, right, unspecified
		<truncated>0</truncated>  # 目标是否被截断，或者被遮挡（超过15%）
		<difficult>0</difficult>  # 检测难易程度，这个主要是根据目标的大小，光照变化，图片质量来判断
		<bndbox>  # 目标位置
			<xmin>263</xmin>
			<ymin>211</ymin>
			<xmax>324</xmax>
			<ymax>339</ymax>
		</bndbox>
	</object>
</annotation>
```