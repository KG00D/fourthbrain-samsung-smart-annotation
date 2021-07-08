// window.onload = function(e) {
//     console.log("loading awesome  ML");
// }

// document.getElementById("formFileSm1").addEventListener("change", (e) => {
//     console.log("file chose " + e.target.files[0]);
// });

function uploadFile1(e) {
    const file = e.target.files[0];
    console.log(file.name, file.size, file.type);
    // fetch("http://0.0.0.0:8081/predict/hello")
    //     .then((response) => {
    //         return response.json()
    //     })
    //     .then((res) => {
    //         console.log(res.response);
    //         //            console.log(`response is ${JSON.parse(res).response}`);
    //     }).catch((e) => {
    //         console.log(e);
    //     })

    const data = new FormData();
    data.append('file', file);

    fetch("http://0.0.0.0:8081/predict/single_image", { method: 'POST', body: data })
        .then((response) => {
            return response.json()
        })
        .then((res) => {
            console.log(res.response, res.size);
        }).catch((e) => {
            console.log(e);
        })

}

function uploadFile2(e) {
    const file = e.target.files[0];
    console.log();
    console.log(file.name, file.size, file.type);

}

function uploadFile3(e) {
    const file = e.target.files[0];
    console.log();
    console.log(file.name, file.size, file.type);

}