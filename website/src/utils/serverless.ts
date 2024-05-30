import axios from 'axios';
import { SITE } from 'astrowind:config';

const getAPIURL = () => {
    return document.location.host.includes("localhost") ? SITE.devAPIURL : "/api"
}

export const sendRequest = async (userId: number, setError, setIsLoading, timeout: number = 180000) => {
    try {
        const res: any = await axios({
            method: "post",
            url: `${getAPIURL()}/recommend`,
            data: { userId },
            timeout
        })
        return res?.data
    } catch (e) {
        setError(e.message)
        setIsLoading(false)
        return null
    }
}