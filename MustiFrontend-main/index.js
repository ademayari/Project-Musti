const url = 'http://127.0.0.1:8000/musti'

async function get_status() {
    const span = document.getElementById('data')
    span.innerText = '...'

    const data = await (await fetch(url)).json()
    
    span.innerText = data.status
}

async function init() {
    get_status()

    const btn = document.getElementById('btn')
    btn.onclick = () => get_status()
}

document.onload = init()