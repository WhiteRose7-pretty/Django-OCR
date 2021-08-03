let image_data;
$(document).ready(function() {
    $(".nav-tabs a").click(function() {
        $(this).tab('show');
    });

    // select image
    $(".thumbnail-images img").click(function() {
        var index = $(this).index();
        $(".thumbnail-images").find('.thumb-active').removeClass('thumb-active');
        $(this).addClass('thumb-active');
        let imagePath = $(this).attr('src');
        $(".image-galleris").find('.gallery-active').addClass('gallery-deactive');
        $(".image-galleris").find('.gallery-active').removeClass('gallery-active');
        $(".image-galleris").find('img:eq(' + index + ')').addClass('gallery-active');
        $(".image-galleris").find('img:eq(' + index + ')').removeClass('gallery-deactive');
        $('#table-data').empty();
        $('#json').empty();
        $('.loading').show();
        getData(imagePath);
    });

    // image upload change event
    $('#upload-image').change(function() {
        let fd = new FormData();
        var files = $(this)[0].files[0];
        var reader = new FileReader();
        $(".image-galleris").find('.gallery-active').addClass('gallery-deactive');
        $(".image-galleris").find('.gallery-active').removeClass('gallery-active');
        $('#blah').addClass('gallery-active');
        $('#blah').removeClass('gallery-deactive');
        reader.onload = function(e) {
            $('#blah').attr('src', e.target.result);
        }
        reader.readAsDataURL(files); // convert to base64 string
        fd.append('file', files);
        $('#table-data').empty();
        $('#json').empty();
        $('.loading').show();
        // image upload to the server
        imageupload(fd);
    });

    // on internet image
    $('#check-image').click(async function() {
        let global_path = $('input[name="search-image"]').val();
        if (global_path == null || global_path == '') {
            alert('Please fill input');
        } else {
            $(".image-galleris").find('.gallery-active').addClass('gallery-deactive');
            $(".image-galleris").find('.gallery-active').removeClass('gallery-active');
            $('#blah').addClass('gallery-active');
            $('#blah').removeClass('gallery-deactive');
            $('#table-data').empty();
            $('#json').empty();
            $('.loading').show();
            $('#myImage').attr('src', global_path);
            $('#blah').attr('src', global_path);
            getBase64FromImageUrl(global_path);
            setTimeout(global_upload, 1000);
        }
    });
});

function global_upload() {
    let json_data = {
        base_image: image_data
    };
    let url_path = image_upload;
    $.post(url_path, json_data, moreResponse);
}

function imageupload(fd) {
    $.ajax({
        url: image_upload,
        type: 'post',
        data: fd,
        contentType: false,
        processData: false,
        success: moreResponse
    });
}

function getData(data) {
    let json_data = {
        path: data
    };
    let url_path = get_detection_data;
    $.post(url_path, json_data, moreResponse);
}

function moreResponse(data, status) {
    console.log(data);
    if (data.status == 0) {
        $('.loading').hide();
        alert(data.data);
    } else {
        var jsonViewer = new JSONViewer();
        document.querySelector("#json").appendChild(jsonViewer.getContainer());
        let result = data['data'];
        console.log(data['data']);
        let table_data = '<table class="table table-bordered"><thead><tr><th>No</th><th>data</th></tr></thead><tbody>';
        result = result.split(' ');
        for (var i = 0; i < result.length; i++) {
            var order = i + 1;
            table_data = table_data + '<tr><th>' + order + '</th><th>' + result[i] + '</th>';

        }
        table_data = table_data + '</tbody></table>';
        jsonViewer.showJSON(data);
        $('.loading').hide();
        $('#table-data').append(table_data);
    }

}

//create base64 image
function getBase64FromImageUrl(url) {
    var img = new Image();
    img.setAttribute('crossOrigin', 'anonymous');
    img.onload = function() {
        var canvas = document.createElement("canvas");
        canvas.width = this.width;
        canvas.height = this.height;

        var ctx = canvas.getContext("2d");
        ctx.drawImage(this, 0, 0);

        var dataURL = canvas.toDataURL("image/png");

        // console.log(dataURL.replace(/^data:image\/(png|jpg);base64,/, ""));
        // image_data = dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
        image_data = dataURL;
    };
    img.src = url;
}