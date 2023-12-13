FROM python:3.8

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các tệp yêu cầu và thư mục vào container
COPY ./app /app/
COPY requirements.txt /app/
COPY model/model.h5 /app/model/

# Cài đặt các gói cần thiết từ tệp requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Chỉ định cổng mà container sẽ lắng nghe
EXPOSE 5000

# Thiết lập biến môi trường
ENV FLASK_APP=app.py

# Chạy ứng dụng khi container được khởi động
CMD ["flask", "run", "--host=0.0.0.0"]
